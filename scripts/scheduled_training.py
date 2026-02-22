import schedule
import time 
import logging
from datetime import datetime
import sys
import os
import traceback


# Add the project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.train import train_all_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def schedule_training(retrain_all=False):
    """
    Periodically check whether SARIMA models need retraining
    and trigger training when required.

    Parameters
    ----------
    retrain_all : bool
        If True, forces retraining of all models immediately
    check_interval_hours : int
        How often to check retraining condition
    """

    logger.info(f"Starting SARIMA training scheduler at {datetime.now()}")

    while True:
        try:
            logger.info("Checking retraining condition...")
            trained_count = train_all_models(retrain_all=retrain_all)
            logger.info(
                f"Training check completed: {trained_count}"
            )
            return trained_count

        except Exception as e:
            logger.error("Training scheduler encountered an error")
            logger.error(str(e))
            logger.error("Full traceback below:")
            logger.error(traceback.format_exc())
            return 0
        
      


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SARIMA model training scheduler"
    )

    parser.add_argument(
        "--retrain-all",
        action="store_true",
        help="Force retraining of all models regardless of last training date"
    )

    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run training once and exit"
    )

    args = parser.parse_args()

    if args.run_once:
        # Run training once and exit
        logger.info("Running SARIMA training once (run-once mode)...")
        try:
            train_all_models(retrain_all=args.retrain_all)
            logger.info("Training completed successfully.")
        except Exception as e:
            logger.error("Error during training")
            logger.error(str(e))
            logger.error("Full traceback:\n" + traceback.format_exc())
    else:
        # Start scheduler for periodic retraining - every 2 weeks
        schedule.every().monday.at("2:30").do(schedule_training)

        logger.info("Scheduled Training started...")
        logger.info("runs every 2 weeks on a Monday at 2.30am.")
        logger.info("Press Ctrl+C to stop.")



        try:
            while True:
                schedule.run_pending()
                time.sleep(60)              # checks every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        