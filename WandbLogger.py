import wandb 

class WandbLogger:
    def __init__(self, entity_name="les-patissiers", project_name="cwasson", config=None, enabled=True, directory="/wandb"):
        self.ENTITY_NAME = entity_name
        self.PROJECT_NAME = project_name
        self.config = config or {}
        self.enabled = enabled

        if self.enabled:
            wandb.init(
                entity=self.ENTITY_NAME,
                project=self.PROJECT_NAME,
                config=self.config,
                dir=directory
            )
            
    def log(self, data):
        if self.enabled:
            wandb.log(data)

    def save(self, path):
        if self.enabled:
            wandb.save(path)