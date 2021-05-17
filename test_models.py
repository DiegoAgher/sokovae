from models.autoencoders import EncoderSmall, DecoderSmall, CnnAE
from models.autoencoders import DeepEncoderSmall, DeepDecoderSmall

encoder = EncoderSmall(9)
decoder = DecoderSmall(9)
ae = CnnAE(encoder, decoder)


print(encoder)
print(decoder)
print(ae)



encoder = DeepEncoderSmall(9)
decoder = DeepDecoderSmall(9)
ae = CnnAE(encoder, decoder)


print(encoder)
print(decoder)
print(ae)
