import os

S3_BUCKET                 = os.environ.get("corp-datainsight-sagemaker-data")
S3_KEY                    = os.environ.get("AKIAJQ7EO4RG5SQCVC4Q")
S3_SECRET                 = os.environ.get("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")
S3_LOCATION               = 'http://{}.s3.amazonaws.com/'.format(S3_BUCKET)

SECRET_KEY                = os.urandom(32)
DEBUG                     = True
PORT                      = 5000