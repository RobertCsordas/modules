import torch
import os
import time
from typing import Optional, List
from collections import defaultdict


class SaverElement:
    def save(self):
        raise NotImplementedError()

    def load(self, saved_state):
        raise NotImplementedError()


class PyObjectSaver(SaverElement):
    def __init__(self, obj):
        self._obj = obj

    def load(self, state):
        def _load(target, state):
            if hasattr(target, "load_state_dict"):
                target.load_state_dict(state)
            elif isinstance(target, dict):
                for k, v in state.items():
                    target[k] = _load(target.get(k), v)
            elif isinstance(target, list):
                if len(target) != len(state):
                    target.clear()
                    for v in state:
                        target.append(v)
                else:
                    for i, v in enumerate(state):
                        target[i] = _load(target[i], v)
            else:
                return state
            return target

        _load(self._obj, state)

    def save(self):
        def _save(target):
            if isinstance(target, (defaultdict, dict)):
                res = target.__class__()
                res.update({k: _save(v) for k, v in target.items()})
            elif hasattr(target, "state_dict"):
                res = target.state_dict()
            elif isinstance(target, list):
                res = [_save(v) for v in target]
            else:
                res = target

            return res

        return _save(self._obj)


class Saver:
    def __init__(self, dir: str, short_interval: int, keep_every_n_hours: int = 4):
        self.savers = {}
        self.short_interval = short_interval
        self.dir = dir
        self._keep_every_n_seconds = keep_every_n_hours * 3600

    def register(self, name: str, saver, replace: bool = False):
        if not replace:
            assert name not in self.savers, "Saver %s already registered" % name

        if isinstance(saver, SaverElement):
            self.savers[name] = saver
        else:
            self.savers[name] = PyObjectSaver(saver)

    def __setitem__(self, key: str, value):
        if value is not None:
            self.register(key, value)

    def save(self, fname: Optional[str] = None, dir: Optional[str] = None, iter: Optional[int]=None):
        state = {}

        if fname is None:
            assert iter is not None, "If fname is not given, iter should be."
            if dir is None:
                dir = self.dir
            fname = os.path.join(dir, self.model_name_from_index(iter))

        os.makedirs(os.path.dirname(fname), exist_ok=True)

        print("Saving %s" % fname)
        for name, fns in self.savers.items():
            state[name] = fns.save()

        try:
            torch.save(state, fname)
        except:
            print("WARNING: Save failed. Maybe running out of disk space?")
            try:
                os.remove(fname)
            except:
                pass
            return None

        return fname

    def tick(self, iter: int):
        if self.short_interval is None or iter % self.short_interval != 0:
            return

        self.save(iter=iter)
        self._cleanup()

    @staticmethod
    def model_name_from_index(index: int) -> str:
        return f"model-{index}.pth"

    @staticmethod
    def get_checkpoint_index_list(dir: str) -> List[int]:
        return list(reversed(sorted(
            [int(fn.split(".")[0].split("-")[-1]) for fn in os.listdir(dir) if fn.split(".")[-1] == "pth"])))

    @staticmethod
    def get_ckpts_in_time_window(dir: str, time_window_s: int, index_list: Optional[List[int]]=None):
        if index_list is None:
            index_list = Saver.get_checkpoint_index_list(dir)

        now = time.time()

        res = []
        for i in index_list:
            name = Saver.model_name_from_index(i)
            mtime = os.path.getmtime(os.path.join(dir, name))
            if now - mtime > time_window_s:
                break

            res.append(name)

        return res

    @staticmethod
    def do_load(fname):
        return torch.load(fname)

    @classmethod
    def load_last_checkpoint(cls, dir: str) -> Optional[any]:
        if not os.path.isdir(dir):
            return None

        last_checkpoint = Saver.get_checkpoint_index_list(dir)

        if last_checkpoint:
            for index in last_checkpoint:
                fname = Saver.model_name_from_index(index)
                try:
                    data = cls.do_load(os.path.join(dir, fname))
                except:
                    continue
                return data
        return None

    def _cleanup(self):
        index_list = self.get_checkpoint_index_list(self.dir)
        new_files = self.get_ckpts_in_time_window(self.dir, self._keep_every_n_seconds, index_list[2:])
        new_files = new_files[:-1]

        for f in new_files:
            os.remove(os.path.join(self.dir, f))

    def load(self, fname=None) -> bool:
        if fname is None:
            state = self.load_last_checkpoint(self.dir)
        else:
            state = self.do_load(fname)

        if not state:
            return False

        # Load all except optimizers
        for k, s in state.items():
            if k not in self.savers:
                print("WARNING: failed to load state of %s. It doesn't exists." % k)
                continue

            if not self.savers[k].load(s):
                return False

        return True
