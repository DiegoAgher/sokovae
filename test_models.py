from models.autoencoders import EncoderSmall, DecoderSmall, CnnAE

encoder = EncoderSmall(9)
decoder = DecoderSmall(9)
ae = CnnAE(encoder, decoder)


print(encoder)
print(decoder)
print(ae)
