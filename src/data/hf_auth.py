"""Hugging Face login from Colab Secrets. The token is never printed, returned or stored by this code."""
import os


def _colab_secret(name):
    from google.colab import userdata

    return userdata.get(name)


def read_secret(getter=None, name="HF_TOKEN"):
    getter = getter or _colab_secret
    try:
        token = getter(name)
    except Exception as error:
        # The type alone says what went wrong (not found, or access not granted).
        raise RuntimeError(
            "Could not read {} from Colab Secrets ({}). Open the key icon in Colab's left "
            "sidebar, add a secret named {}, and allow this notebook to access it.".format(
                name, type(error).__name__, name
            )
        ) from None
    if not token or not token.strip():
        raise RuntimeError(
            "HF_TOKEN is missing. Add it through Google Colab Secrets (the key icon in the sidebar)."
        )
    return token.strip()


def authenticate(getter=None, login_fn=None, forbidden_root=None):
    """Log in without saving to Git credentials. Returns nothing, so no token reaches notebook output.

    login() keeps a copy at HF_TOKEN_PATH, which follows HF_HOME. Refuse to continue if that
    path is on Drive, because the token must stay on Colab's temporary disk.
    """
    from huggingface_hub import constants

    forbidden_root = forbidden_root or os.environ.get("FS2_DRIVE_ROOT") or "/content/drive"
    token_path = os.path.abspath(constants.HF_TOKEN_PATH)
    if token_path.startswith(os.path.abspath(forbidden_root)):
        raise RuntimeError("HF_HOME points at Google Drive; the token would be saved there. Unset HF_HOME.")

    token = read_secret(getter)
    if login_fn is None:
        from huggingface_hub import login as login_fn
    try:
        login_fn(token=token, add_to_git_credential=False)
    except Exception as error:
        raise RuntimeError(
            "Hugging Face rejected the login ({}). Check that HF_TOKEN is a valid Read token.".format(
                type(error).__name__
            )
        ) from None
    print("Hugging Face authentication succeeded.")
