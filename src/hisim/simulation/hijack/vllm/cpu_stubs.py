"""CUDA event shim; modeled transfer deadlines are tracked separately."""


class DummyEvent:
    """Report immediate completion without CUDA synchronization."""

    def record(self, stream=None):
        pass

    def wait(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0
