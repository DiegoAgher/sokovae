from models.autoencoders import EncoderSmall, DecoderSmall, CnnAE
from models.autoencoders import DeepEncoderSmall, DeepDecoderSmall
from models.autoencoders import SmallEncoder40, SmallDecoder40

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


encoder = SmallEncoder40 (9, [32, 64, 128], non_variational=True)
decoder = SmallDecoder40(9,  list(reversed([32, 64, 128])))
ae = CnnAE(encoder, decoder)
