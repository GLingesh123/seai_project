from data.loaders.stream_loader import StreamLoader
from models.ssl.autoencoder import SSLAutoencoder
from models.ssl.ssl_trainer import SSLTrainer

stream = StreamLoader(scenario={"type": "none"})
model = SSLAutoencoder()
trainer = SSLTrainer(model)

trainer.pretrain_stream(stream, steps=60)

print(trainer.summary())
