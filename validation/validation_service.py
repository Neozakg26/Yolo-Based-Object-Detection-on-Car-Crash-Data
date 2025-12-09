class ValidationService:
    def __init__(self, selected_strategy, config, metric_calculator):
        self.selected_strategy = selected_strategy
        self.config = config
        self.metric_calculator = metric_calculator

    def validate(self, model, observers):
        results = self.selected_strategy.validate(model, self.config, observers)

        return self.metric_calculator.compute(results) #predictions, targets)
