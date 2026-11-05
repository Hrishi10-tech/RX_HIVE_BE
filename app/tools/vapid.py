"""Generate a VAPID keypair for Web Push. Run: python -m app.tools.vapid"""

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def main() -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    private_raw = key.private_numbers().private_value.to_bytes(32, "big")
    public_point = key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    print("RXHIVE_VAPID_PUBLIC_KEY=" + _b64(public_point))
    print("RXHIVE_VAPID_PRIVATE_KEY=" + _b64(private_raw))
    print("RXHIVE_VAPID_SUBJECT=mailto:admin@rhythmrx.ai")


if __name__ == "__main__":
    main()
