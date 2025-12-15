class TrainerService:
    def __init__(self, selected_strategy, train_config_builder):
        self.selected_strategy = selected_strategy
        self.train_config_builder = train_config_builder

    def validate_env(self):
        return self.selected_strategy.validate_environment()
    
    def train(self, model, config, observers):
        train_cfg = self.train_config_builder.build(config)
        return self.selected_strategy.train(model, train_cfg, observers)
