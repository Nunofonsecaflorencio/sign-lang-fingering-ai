# Sign Language Fingering Detection AI

## Pre-Requisites

1. Create an environment

```powershell
python -m venv env
```

2. Activate the environment (in the env)

For PowerShell

```
.\env\Scripts\activate.ps1
```

For "Other" CLI

```
.\env\Scripts\activate
```

3. Install Requirements

```
pip install -r requirements.txt
```

## To collect data

```
python collect_data.py
```

## To Train Model

Run `SignFingeringDetection.ipynb`

## To Test

```
python realtime_detector.py
```
