from .validation_strat_impl import SimpleValidationStrategy

class ValidationStrategySelector:
    @staticmethod
    def select(name):
        if name == "simple":
            return SimpleValidationStrategy()
        raise ValueError(f"Unknown validation strategy: {name}")
