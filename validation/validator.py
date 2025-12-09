class Validator:
    def __init__(self,model, validation_service):
        self.model = model
        self.validation_service = validation_service

    def run(self, observers):
        return self.validation_service.validate(self.model, observers)
