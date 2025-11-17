#!/bin/bash

echo "Starting IRIS for Health and setting up FHIR..."

# Start IRIS
iris start IRIS

# Enable FHIR via InterSystems terminal commands
iris session IRIS <<EOF
zn "FHIRSERVER"
do ##class(HS.FHIRServer.Tools.Install).InstallFHIRServer("FHIRDemo", "/fhirr4")
halt
EOF

# Restart IRIS to apply changes
iris stop IRIS quietly
iris start IRIS