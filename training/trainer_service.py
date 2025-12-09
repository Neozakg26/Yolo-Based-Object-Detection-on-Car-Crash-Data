class TrainerService:
    def __init__(self, strategy_selector, train_config_builder):
        self.strategy_selector = strategy_selector
        self.train_config_builder = train_config_builder

    def train(self, model, config, observers):
        train_cfg = self.train_config_builder.build(config)
        strategy = self.strategy_selector.select(config.compute["strategy"])
        return strategy.train(model, train_cfg, observers)
