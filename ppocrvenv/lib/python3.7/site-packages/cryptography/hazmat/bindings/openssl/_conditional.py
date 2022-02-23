# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


def cryptography_has_ec2m():
    return [
        "EC_POINT_get_affine_coordinates_GF2m",
    ]


def cryptography_has_rsa_oaep_md():
    return [
        "EVP_PKEY_CTX_set_rsa_oaep_md",
    ]


def cryptography_has_rsa_oaep_label():
    return [
        "EVP_PKEY_CTX_set0_rsa_oaep_label",
    ]


def cryptography_has_ssl3_method():
    return [
        "SSLv3_method",
        "SSLv3_client_method",
        "SSLv3_server_method",
    ]


def cryptography_has_110_verification_params():
    return ["X509_CHECK_FLAG_NEVER_CHECK_SUBJECT"]


def cryptography_has_set_cert_cb():
    return [
        "SSL_CTX_set_cert_cb",
        "SSL_set_cert_cb",
    ]


def cryptography_has_ssl_st():
    return [
        "SSL_ST_BEFORE",
        "SSL_ST_OK",
        "SSL_ST_INIT",
        "SSL_ST_RENEGOTIATE",
    ]


def cryptography_has_tls_st():
    return [
        "TLS_ST_BEFORE",
        "TLS_ST_OK",
    ]


def cryptography_has_scrypt():
    return [
        "EVP_PBE_scrypt",
    ]


def cryptography_has_evp_pkey_dhx():
    return [
        "EVP_PKEY_DHX",
    ]


def cryptography_has_mem_functions():
    return [
        "Cryptography_CRYPTO_set_mem_functions",
    ]


def cryptography_has_x509_store_ctx_get_issuer():
    return [
        "X509_STORE_get_get_issuer",
        "X509_STORE_set_get_issuer",
    ]


def cryptography_has_ed448():
    return [
        "EVP_PKEY_ED448",
        "NID_ED448",
    ]


def cryptography_has_ed25519():
    return [
        "NID_ED25519",
        "EVP_PKEY_ED25519",
    ]


def cryptography_has_poly1305():
    return [
        "NID_poly1305",
        "EVP_PKEY_POLY1305",
    ]


def cryptography_has_oneshot_evp_digest_sign_verify():
    return [
        "EVP_DigestSign",
        "EVP_DigestVerify",
    ]


def cryptography_has_evp_digestfinal_xof():
    return [
        "EVP_DigestFinalXOF",
    ]


def cryptography_has_evp_pkey_get_set_tls_encodedpoint():
    return [
        "EVP_PKEY_get1_tls_encodedpoint",
        "EVP_PKEY_set1_tls_encodedpoint",
    ]


def cryptography_has_fips():
    return [
        "FIPS_mode_set",
        "FIPS_mode",
    ]


def cryptography_has_psk():
    return [
        "SSL_CTX_use_psk_identity_hint",
        "SSL_CTX_set_psk_server_callback",
        "SSL_CTX_set_psk_client_callback",
    ]


def cryptography_has_custom_ext():
    return [
        "SSL_CTX_add_client_custom_ext",
        "SSL_CTX_add_server_custom_ext",
        "SSL_extension_supported",
    ]


def cryptography_has_openssl_cleanup():
    return [
        "OPENSSL_cleanup",
    ]


def cryptography_has_tlsv13():
    return [
        "TLS1_3_VERSION",
        "SSL_OP_NO_TLSv1_3",
    ]


def cryptography_has_tlsv13_functions():
    return [
        "SSL_VERIFY_POST_HANDSHAKE",
        "SSL_CTX_set_ciphersuites",
        "SSL_verify_client_post_handshake",
        "SSL_CTX_set_post_handshake_auth",
        "SSL_set_post_handshake_auth",
        "SSL_SESSION_get_max_early_data",
        "SSL_write_early_data",
        "SSL_read_early_data",
        "SSL_CTX_set_max_early_data",
    ]


def cryptography_has_keylog():
    return [
        "SSL_CTX_set_keylog_callback",
        "SSL_CTX_get_keylog_callback",
    ]


def cryptography_has_raw_key():
    return [
        "EVP_PKEY_new_raw_private_key",
        "EVP_PKEY_new_raw_public_key",
        "EVP_PKEY_get_raw_private_key",
        "EVP_PKEY_get_raw_public_key",
    ]


def cryptography_has_engine():
    return [
        "ENGINE_by_id",
        "ENGINE_init",
        "ENGINE_finish",
        "ENGINE_get_default_RAND",
        "ENGINE_set_default_RAND",
        "ENGINE_unregister_RAND",
        "ENGINE_ctrl_cmd",
        "ENGINE_free",
        "ENGINE_get_name",
        "Cryptography_add_osrandom_engine",
        "ENGINE_ctrl_cmd_string",
        "ENGINE_load_builtin_engines",
        "ENGINE_load_private_key",
        "ENGINE_load_public_key",
        "SSL_CTX_set_client_cert_engine",
    ]


def cryptography_has_verified_chain():
    return [
        "SSL_get0_verified_chain",
    ]


def cryptography_has_srtp():
    return [
        "SSL_CTX_set_tlsext_use_srtp",
        "SSL_set_tlsext_use_srtp",
        "SSL_get_selected_srtp_profile",
    ]


def cryptography_has_get_proto_version():
    return [
        "SSL_CTX_get_min_proto_version",
        "SSL_CTX_get_max_proto_version",
        "SSL_get_min_proto_version",
        "SSL_get_max_proto_version",
    ]


def cryptography_has_providers():
    return [
        "OSSL_PROVIDER_load",
        "OSSL_PROVIDER_unload",
        "ERR_LIB_PROV",
        "PROV_R_WRONG_FINAL_BLOCK_LENGTH",
        "PROV_R_BAD_DECRYPT",
    ]


def cryptography_has_op_no_renegotiation():
    return [
        "SSL_OP_NO_RENEGOTIATION",
    ]


def cryptography_has_dtls_get_data_mtu():
    return [
        "DTLS_get_data_mtu",
    ]


def cryptography_has_300_fips():
    return [
        "EVP_default_properties_is_fips_enabled",
        "EVP_default_properties_enable_fips",
    ]


def cryptography_has_ssl_cookie():
    return [
        "SSL_OP_COOKIE_EXCHANGE",
        "DTLSv1_listen",
        "SSL_CTX_set_cookie_generate_cb",
        "SSL_CTX_set_cookie_verify_cb",
    ]


def cryptography_has_pkcs7_funcs():
    return [
        "SMIME_write_PKCS7",
        "PEM_write_bio_PKCS7_stream",
        "PKCS7_sign_add_signer",
        "PKCS7_final",
        "PKCS7_verify",
        "SMIME_read_PKCS7",
        "PKCS7_get0_signers",
    ]


def cryptography_has_bn_flags():
    return [
        "BN_FLG_CONSTTIME",
        "BN_set_flags",
        "BN_prime_checks_for_size",
    ]


def cryptography_has_evp_pkey_dh():
    return [
        "EVP_PKEY_set1_DH",
    ]


# This is a mapping of
# {condition: function-returning-names-dependent-on-that-condition} so we can
# loop over them and delete unsupported names at runtime. It will be removed
# when cffi supports #if in cdef. We use functions instead of just a dict of
# lists so we can use coverage to measure which are used.
CONDITIONAL_NAMES = {
    "Cryptography_HAS_EC2M": cryptography_has_ec2m,
    "Cryptography_HAS_RSA_OAEP_MD": cryptography_has_rsa_oaep_md,
    "Cryptography_HAS_RSA_OAEP_LABEL": cryptography_has_rsa_oaep_label,
    "Cryptography_HAS_SSL3_METHOD": cryptography_has_ssl3_method,
    "Cryptography_HAS_110_VERIFICATION_PARAMS": (
        cryptography_has_110_verification_params
    ),
    "Cryptography_HAS_SET_CERT_CB": cryptography_has_set_cert_cb,
    "Cryptography_HAS_SSL_ST": cryptography_has_ssl_st,
    "Cryptography_HAS_TLS_ST": cryptography_has_tls_st,
    "Cryptography_HAS_SCRYPT": cryptography_has_scrypt,
    "Cryptography_HAS_EVP_PKEY_DHX": cryptography_has_evp_pkey_dhx,
    "Cryptography_HAS_MEM_FUNCTIONS": cryptography_has_mem_functions,
    "Cryptography_HAS_X509_STORE_CTX_GET_ISSUER": (
        cryptography_has_x509_store_ctx_get_issuer
    ),
    "Cryptography_HAS_ED448": cryptography_has_ed448,
    "Cryptography_HAS_ED25519": cryptography_has_ed25519,
    "Cryptography_HAS_POLY1305": cryptography_has_poly1305,
    "Cryptography_HAS_ONESHOT_EVP_DIGEST_SIGN_VERIFY": (
        cryptography_has_oneshot_evp_digest_sign_verify
    ),
    "Cryptography_HAS_EVP_PKEY_get_set_tls_encodedpoint": (
        cryptography_has_evp_pkey_get_set_tls_encodedpoint
    ),
    "Cryptography_HAS_FIPS": cryptography_has_fips,
    "Cryptography_HAS_PSK": cryptography_has_psk,
    "Cryptography_HAS_CUSTOM_EXT": cryptography_has_custom_ext,
    "Cryptography_HAS_OPENSSL_CLEANUP": cryptography_has_openssl_cleanup,
    "Cryptography_HAS_TLSv1_3": cryptography_has_tlsv13,
    "Cryptography_HAS_TLSv1_3_FUNCTIONS": cryptography_has_tlsv13_functions,
    "Cryptography_HAS_KEYLOG": cryptography_has_keylog,
    "Cryptography_HAS_RAW_KEY": cryptography_has_raw_key,
    "Cryptography_HAS_EVP_DIGESTFINAL_XOF": (
        cryptography_has_evp_digestfinal_xof
    ),
    "Cryptography_HAS_ENGINE": cryptography_has_engine,
    "Cryptography_HAS_VERIFIED_CHAIN": cryptography_has_verified_chain,
    "Cryptography_HAS_SRTP": cryptography_has_srtp,
    "Cryptography_HAS_GET_PROTO_VERSION": cryptography_has_get_proto_version,
    "Cryptography_HAS_PROVIDERS": cryptography_has_providers,
    "Cryptography_HAS_OP_NO_RENEGOTIATION": (
        cryptography_has_op_no_renegotiation
    ),
    "Cryptography_HAS_DTLS_GET_DATA_MTU": cryptography_has_dtls_get_data_mtu,
    "Cryptography_HAS_300_FIPS": cryptography_has_300_fips,
    "Cryptography_HAS_SSL_COOKIE": cryptography_has_ssl_cookie,
    "Cryptography_HAS_PKCS7_FUNCS": cryptography_has_pkcs7_funcs,
    "Cryptography_HAS_BN_FLAGS": cryptography_has_bn_flags,
    "Cryptography_HAS_EVP_PKEY_DH": cryptography_has_evp_pkey_dh,
}
