#!/bin/bash
echo Generatin the list of all files in the AWS S3 bucket noaa-hrrr-bdp-pds
aws s3 ls s3://noaa-hrrr-bdp-pds/ --no-sign-request --recursive >& hrrr-all.txt
