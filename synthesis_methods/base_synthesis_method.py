class BaseSynthesisMethod:
    def __init__(self, *args, **kwargs):
        pass
    def synthesis(self, *args, **kwargs):
        if self.pre_synthesis_hook_enabled():
            self.pre_synthesis_method(*args, **kwargs)
        ret = self.impl(*args, **kwargs)
        if self.post_synthesis_hook_enabled():
            self.post_synthesis_method(*args, **kwargs)
        return ret
    def pre_synthesis_method(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")
    def pre_synthesis_hook_enabled(self):
        return True
    def post_synthesis_method(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")
    def post_synthesis_hook_enabled(self):
        return True
    def impl(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses")