"""Utility function or classes for syncers."""
import threading


_collective_keys = None
_collective_keys_lock = threading.Lock()


def get_collective_keys():
    """Return a singleton instance of CollectiveKey."""
    global _collective_keys
    if _collective_keys:
        return _collective_keys
    _collective_keys_lock.acquire()

    try:
        if _collective_keys:
            return _collective_keys
        collective_keys = CollectiveKey()
        _collective_keys = collective_keys
        return _collective_keys
    finally:
        _collective_keys_lock.release()


class CollectiveKey:
    """A hash that generates group key and instance key for allreduce."""

    def __init__(self,
                 group_key_start=1,
                 instance_key_start=1000):
        """Init the collective key."""
        self._group_key = group_key_start
        self._group_key_dict = {}
        self._instance_key = instance_key_start
        self._instance_key_dict = {}

    def get_group_key(self, canonical_devices):
        """Generate or retrieve the group key based on a list of strings of the participating devices."""
        for d in canonical_devices:
            if not isinstance(d, str):
                raise ValueError('Need canonicalized devices')
        key_id = ','.join(canonical_devices)
        if key_id not in self._group_key_dict:
            new_key = self._group_key
            self._group_key += 1
            self._group_key_dict[key_id] = new_key
        return self._group_key_dict[key_id]

    def get_instance_key(self, var_name):
        """Generate or retrieve the instance key based on the *original* variable name."""
        key_id = var_name
        if key_id not in self._instance_key_dict:
            new_key = self._instance_key
            self._instance_key += 1
            self._instance_key_dict[key_id] = new_key
        return self._instance_key_dict[key_id]