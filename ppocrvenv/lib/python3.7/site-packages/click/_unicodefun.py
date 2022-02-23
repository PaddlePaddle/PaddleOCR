import codecs
import os
from gettext import gettext as _


def _verify_python_env() -> None:
    """Ensures that the environment is good for Unicode."""
    try:
        from locale import getpreferredencoding

        fs_enc = codecs.lookup(getpreferredencoding()).name
    except Exception:
        fs_enc = "ascii"

    if fs_enc != "ascii":
        return

    extra = [
        _(
            "Click will abort further execution because Python was"
            " configured to use ASCII as encoding for the environment."
            " Consult https://click.palletsprojects.com/unicode-support/"
            " for mitigation steps."
        )
    ]

    if os.name == "posix":
        import subprocess

        try:
            rv = subprocess.Popen(
                ["locale", "-a"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="ascii",
                errors="replace",
            ).communicate()[0]
        except OSError:
            rv = ""

        good_locales = set()
        has_c_utf8 = False

        for line in rv.splitlines():
            locale = line.strip()

            if locale.lower().endswith((".utf-8", ".utf8")):
                good_locales.add(locale)

                if locale.lower() in ("c.utf8", "c.utf-8"):
                    has_c_utf8 = True

        if not good_locales:
            extra.append(
                _(
                    "Additional information: on this system no suitable"
                    " UTF-8 locales were discovered. This most likely"
                    " requires resolving by reconfiguring the locale"
                    " system."
                )
            )
        elif has_c_utf8:
            extra.append(
                _(
                    "This system supports the C.UTF-8 locale which is"
                    " recommended. You might be able to resolve your"
                    " issue by exporting the following environment"
                    " variables:"
                )
            )
            extra.append("    export LC_ALL=C.UTF-8\n    export LANG=C.UTF-8")
        else:
            extra.append(
                _(
                    "This system lists some UTF-8 supporting locales"
                    " that you can pick from. The following suitable"
                    " locales were discovered: {locales}"
                ).format(locales=", ".join(sorted(good_locales)))
            )

        bad_locale = None

        for env_locale in os.environ.get("LC_ALL"), os.environ.get("LANG"):
            if env_locale and env_locale.lower().endswith((".utf-8", ".utf8")):
                bad_locale = env_locale

            if env_locale is not None:
                break

        if bad_locale is not None:
            extra.append(
                _(
                    "Click discovered that you exported a UTF-8 locale"
                    " but the locale system could not pick up from it"
                    " because it does not exist. The exported locale is"
                    " {locale!r} but it is not supported."
                ).format(locale=bad_locale)
            )

    raise RuntimeError("\n\n".join(extra))
