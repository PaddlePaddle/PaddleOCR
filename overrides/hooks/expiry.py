import re
from datetime import datetime


def on_page_context(context, page, config, nav):
    expiry_days = config.get("extra", {}).get("expiry_days", 365)

    def compute_expiry(meta):
        revision = (
            meta.get("git_revision_date_localized")
            or meta.get("git_creation_date_localized")
            or meta.get("revision_date")
        )
        is_expired = False
        last_update = None
        if revision:
            m = re.search(r"(\d{4}-\d{2}-\d{2})", str(revision))
            if m:
                last_update = m.group(1)
                try:
                    dt = datetime.strptime(last_update, "%Y-%m-%d")
                    if (datetime.now() - dt).days > expiry_days:
                        is_expired = True
                except Exception:
                    # 无法解析日期时，保持不显示过期提示
                    pass
        return is_expired, last_update

    page.is_expired, page.last_update = compute_expiry(page.meta)
    context["is_expired"] = page.is_expired
    context["last_update"] = page.last_update
    context["expiry_days"] = expiry_days

    return context
