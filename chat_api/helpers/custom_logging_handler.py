from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud import error_reporting
from google.cloud import logging as google_logging
import logging

reporting_client = error_reporting.Client()


class CustomErrorReportingGCloudHandler(google_logging.handlers.handlers.CloudLoggingHandler):
    def emit(self, record):
        super().emit(record)

        if record.levelno >= logging.ERROR:
            if record.exc_info:
                reporting_client.report_exception()
            else:
                message = super(CloudLoggingHandler, self).format(record)
                reporting_client.report(message)
