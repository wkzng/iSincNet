#!/bin/bash
ARCHIVE=sincnet.zip
echo "Packaging SageMaker Models for External Training!"
zip -r ${ARCHIVE} datasets sincnet training requirements.txt
echo "Packaging Done"