import logging
import time
from typing import Optional

import requests

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def ping(url: str = "https://apidash.azurewebsites.net/", sleep_secs: Optional[float] = 600):

    count = 1
    while count == 1 or sleep_secs:
        logging.info("Request %s sent to %s", count, url)

        response = requests.get(url)
        logging.info(
            "Response %s received, status_code=%s, elapsed=%s",
            count,
            response.status_code,
            response.elapsed,
        )
        text_len = len(response.text)
        logging.info(
            "Response text %s received, len(text)=%s, elapsed=%s",
            count,
            text_len,
            response.elapsed,
        )

        count += 1
        logging.info("Sleeping %s seconds", sleep_secs)
        if sleep_secs:
            time.sleep(sleep_secs)  # type: ignore


if __name__ == "__main__":
    ping(sleep_secs=60)