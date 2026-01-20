from dataclasses import dataclass

from portfolio_agent.config.settings import get_settings

@dataclass
class RHConfig:
    username: str | None
    password: str | None
    totp: str | None
    use_live: bool
    csv_path: str | None

def load_config() -> RHConfig:
    settings = get_settings()
    return RHConfig(
        username=settings.robinhood_username,
        password=settings.robinhood_password,
        totp=settings.robinhood_totp,
        use_live=settings.robinhood_use_live,
        csv_path=settings.robinhood_csv_path,
    )
