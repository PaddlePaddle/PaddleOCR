from Crypto.PublicKey.RSA import RsaKey

from Crypto.Signature.pkcs1_15 import PKCS115_SigScheme


def new(rsa_key: RsaKey) -> PKCS115_SigScheme: ...