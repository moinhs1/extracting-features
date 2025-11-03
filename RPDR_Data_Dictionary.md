# RPDR Data Dictionary
## Research Patient Data Registry Files

**Generated Date:** 2025-08-08 10:11:13
**Directory:** /media/moin/Elements/rpdr
**Total Files:** 25

---

## Overview

The Research Patient Data Registry (RPDR) is the Partners HealthCare system's centralized clinical data warehouse. This data dictionary documents all RPDR export files and their contents.

## File Summary

| File Suffix | Description | Columns | Purpose |
|-------------|-------------|---------|---------|
| **Bib** | Billing/Insurance - Insurance billing codes and related information | 14 |  Insurance billing codes and related information |
| **Car** | Cardiology - Cardiac-specific data including echocardiograms, catheterizations | 10 |  Cardiac |
| **Dem** | Demographics - Patient demographic information | 22 |  Patient demographic information |
| **Dia** | Diagnoses - ICD diagnosis codes and descriptions | 14 |  ICD diagnosis codes and descriptions |
| **Dis** | Discharge - Discharge summaries and related notes | 10 |  Discharge summaries and related notes |
| **Enc** | Encounters - Hospital encounters and visits | 32 |  Hospital encounters and visits |
| **Hnp** | History & Physical - H&P notes and examinations | 10 |  H&P notes and examinations |
| **Lab** | Laboratory - Lab test results and values | 22 |  Lab test results and values |
| **Lno** | Lab Notes - Laboratory notes and comments | 13 |  Laboratory notes and comments |
| **Log** | Log - System log entries | 1 |  System log entries |
| **Med** | Medications - Medication prescriptions and administrations | 16 |  Medication prescriptions and administrations |
| **Mic** | Microbiology - Microbiology culture results | 16 |  Microbiology culture results |
| **Mrn** | Medical Record Numbers - Patient MRN mappings | 15 |  Patient MRN mappings |
| **Opn** | Operative Notes - Surgical and procedure notes | 10 |  Surgical and procedure notes |
| **Pat** | Pathology - Pathology reports and results | 10 |  Pathology reports and results |
| **Phy** | Physicians - Provider information | 15 |  Provider information |
| **Prc** | Procedures - CPT procedure codes | 15 |  CPT procedure codes |
| **Prg** | Progress Notes - Clinical progress notes | 10 |  Clinical progress notes |
| **Pul** | Pulmonary - Pulmonary function tests | 10 |  Pulmonary function tests |
| **Rad** | Radiology - Radiology reports and imaging results | 10 |  Radiology reports and imaging results |
| **Rdt** | Radiology Details - Additional radiology information | 14 |  Additional radiology information |
| **Rfv** | Reason for Visit - Chief complaints and visit reasons | 13 |  Chief complaints and visit reasons |
| **Trn** | Transfusions - Blood product transfusions | 11 |  Blood product transfusions |
| **Vis** | Vital Signs - Vital sign measurements | 10 |  Vital sign measurements |


---

## Detailed File Specifications

### Bib - Billing/Insurance - Insurance billing codes and related information

**File:** `FNR_20240409_091633_Bib.txt`

**Total Columns:** 14

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| Registry_Name | object | Name field | 0 | 1 | `Biobank` |
| Subject_Id | int64 | Identifier field | 0 | 1000 | `10050935`, `10045207`, `10046626` |
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 1000 | `102878749`, `103436077`, `102699174` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 1000 | `10065763988`, `10043210078`, `10030193246` |
| MGH_MRN | float64 | Massachusetts General Hospital MRN | 17 | 983 | `2383044.0`, `6845413.0`, `5765583.0` |
| BWH_MRN | float64 | Brigham and Women's Hospital MRN | 40 | 960 | `3983053.0`, `30784979.0`, `47689724.0` |
| FH_MRN | float64 | Field specific to Bib file | 385 | 615 | `10496784.0`, `1085425.0`, `686450.0` |
| SRH_MRN | float64 | Field specific to Bib file | 588 | 412 | `100175164.0`, `414930.0`, `87679.0` |
| NWH_MRN | float64 | Field specific to Bib file | 188 | 812 | `11066916.0`, `11419156.0`, `774287.0` |
| NSMC_MRN | float64 | Field specific to Bib file | 623 | 377 | `10907264.0`, `404438.0`, `10576062.0` |
| MCL_MRN | float64 | Field specific to Bib file | 904 | 96 | `280218.0`, `282844.0`, `108329.0` |
| MEE_MRN | float64 | Field specific to Bib file | 329 | 671 | `9879858.0`, `7386023.0`, `7881611.0` |
| DFC_MRN | float64 | Field specific to Bib file | 338 | 662 | `646337.0`, `735343.0`, `897302.0` |
| WDH_MRN | float64 | Field specific to Bib file | 912 | 88 | `847113.0`, `831892.0`, `953983.0` |

---

### Car - Cardiology - Cardiac-specific data including echocardiograms, catheterizations

**File:** `FNR_20240409_091633_Car.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 711 | `100003884`, `ATRIAL RATE 66 BPM`, `PR INTERVAL 162 ms` |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 976 | 3 | `10040029737.0`, `10033842823.0`, `10040017377.0` |
| MRN_Type | object | Field specific to Car file | 976 | 2 | `BWH`, `MGH` |
| MRN | float64 | Field specific to Car file | 976 | 3 | `667360.0`, `782771.0`, `8054.0` |
| Report_Number | float64 | Identifier field | 976 | 24 | `925343615.0`, `969816804.0`, `969816797.0` |
| Report_Date_Time | object | Date field | 976 | 24 | `2/18/2022 2:56:38 PM`, `3/29/2022 7:49:04 AM`, `3/23/2022 12:56:08 PM` |
| Report_Description | object | Field specific to Car file | 976 | 602 | `ECG 12-LEAD (Test:ECG1)`, `ECHO COMPLETE TTE (Test:ECH104)`, `NC PET CARDIAC PERFUSION MULTIPLE (Test:NC.PET.PRFMULTI)`, `US LOWER EXTREMITY VEINS DUPLEX (BILATERAL)`, `CARDIAC CATHETERIZATION (Test:CATH01)`, `ECHO TRANSESOPHAGEAL COMPLETE`, `EP STUDY`, `EP DEVICE CHECK / FOLLOW UP`, `NC MYOCARDIAL PERFUSION PHARMACOLOGIC STRESS MULTIPLE`, `NC MYOCARDIAL PERFUSION EXERCISE, MULTI`, `MCT (MOBILE CARDIAC TELEMETRY)`, `US UPPER EXTREMITY VEINS DUPLEX (BILATERAL)`, `CARDIOVERSION`, `NC MYOCARDIAL PERFUSION STRESS SINGLE`, `US LOWER EXTREMITY VEINS MAPPING`, `STRESS TEST EXERCISE`, `INTRAOPERATIVE ECHO TRANSESOPHAGEAL` (20 of 602 unique values) |
| Report_Status | object | Status of report (final, preliminary, etc.) | 976 | 19 | `F`, `R`, `Final`, `Signed`, `Archived`, `Revised Final`, `Canceled`, `Revised`, `Complete`, `Preliminary`, `Draft`, `Scheduled`, `Appended/Signed`, `Cancelled`, `Pending`, `I`, `Completed`, `Ordered`, `PA` |
| Report_Type | object | Field specific to Car file | 976 | 150 | `BCAR:ECG1`, `BCAR:ECH104`, `BCAR:NC.PET.PRFMULTI`, `BCAR:US.VA.VVELEX/B`, `MCAR:ECG1`, `MCAR:ECH104`, `MCAR:CATH01`, `NCAR:ECG1`, `BCAR:ECH200`, `BCAR:EP22`, `MCAR:EP518`, `WCAR:ECG1`, `FCAR:ECG1`, `FCAR:ECH104`, `FCAR:ECH200`, `MCAR:NC.NC.PRFPHRMM`, `WCAR:NC.NC.PRFEXERM`, `WCAR:ECH104`, `MCAR:CAR103`, `BCAR:CATH01` (20 of 150 unique values) |
| Report_Text | object | Full text of report | 976 | 12 | `VENTRICULAR RATE EKG/MIN 66 BPM`, `Result Report `, `VENTRICULAR RATE EKG/MIN 69 BPM` |

---

### Dem - Demographics - Patient demographic information

**File:** `FNR_20240409_091633_Dem.txt`

**Total Columns:** 22

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 1000 | `100004212`, `100003884`, `100004796` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 1000 | `10040032343`, `10040029737`, `10033842823` |
| MRN_Type | object | Field specific to Dem file | 0 | 166 | `PMRN, MGH, BWH, NSM, MEE`, `PMRN, MGH, BWH, FH, NWH, MEE, DFC`, `PMRN, MGH, BWH, FH, NWH, NSM` |
| MRN | object | Field specific to Dem file | 0 | 1000 | `10040032343, 0038696, 09556150, 00364545, 7327588`, `10040029737, 4788338, 00667360, 00662134, 1157... |
| Gender_Legal_Sex | object | Field specific to Dem file | 0 | 2 | `Male`, `Female` |
| Date_of_Birth | object | Patient date of birth | 0 | 978 | `05/29/1931`, `02/07/1945`, `04/24/1931` |
| Age | int64 | Field specific to Dem file | 0 | 77 | `88`, `79`, `92` |
| Sex_At_Birth | object | Field specific to Dem file | 0 | 3 | `Unknown`, `Female`, `Male` |
| Gender_Identity | object | Identifier field | 0 | 5 | `Unknown`, `Female`, `Male` |
| Language | object | Field specific to Dem file | 0 | 20 | `English`, `Spanish`, `Unknown` |
| Language_group | object | Field specific to Dem file | 0 | 3 | `English`, `Non-English`, `Unknown` |
| Race1 | object | Field specific to Dem file | 0 | 9 | `White`, `Black`, `Native Hawaiian or Other Pacific Islander` |
| Race2 | object | Field specific to Dem file | 994 | 2 | `White`, `Other` |
| Race_Group | object | Field specific to Dem file | 0 | 8 | `White`, `Black`, `Two or More` |
| Ethnic_Group | object | Patient ethnicity | 0 | 4 | `Non Hispanic`, `DECLINED`, `Unknown/Missing` |
| Marital_status | object | Field specific to Dem file | 0 | 8 | `Widowed`, `Single`, `Married` |
| Religion | object | Field specific to Dem file | 6 | 33 | `Catholic`, `Jewish`, `Protestant` |
| Is_a_veteran | object | Field specific to Dem file | 0 | 7 | `No`, `Yes`, `NO, NVR SERV` |
| Zip_code | int64 | Code identifier | 0 | 290 | `2151`, `2120`, `2467` |
| Country | object | Field specific to Dem file | 0 | 3 | `United States`, `Not recorded`, `China` |
| Vital_status | object | Field specific to Dem file | 0 | 3 | `Date of Death reported from a Partners Hospital`, `Not reported as deceased`, `Date of Death report... |
| Date_Of_Death | object | Date field | 473 | 438 | `12/07/2019`, `05/13/2012`, `12/18/2011` |

---

### Dia - Diagnoses - ICD diagnosis codes and descriptions

**File:** `FNR_20240409_091633_Dia.txt`

**Total Columns:** 14

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 69 | `100071500`, `100011188`, `100071228` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 69 | `10042818004`, `10040154543`, `10024374380` |
| MRN_Type | object | Field specific to Dia file | 0 | 6 | `MGH`, `BWH`, `MEE` |
| MRN | int64 | Field specific to Dia file | 0 | 81 | `881883`, `1381532`, `3448615` |
| Date | object | Date field | 0 | 713 | `7/16/2012`, `4/24/2014`, `12/19/2012` |
| Diagnosis_Name | object | Diagnosis description | 0 | 106 | `Pulmonary diseases due to other mycobacteria`, `Unspecified septicemia`, `Herpes zoster without men... |
| Code_Type | object | Type of code (ICD-9, ICD-10, CPT, etc.) | 0 | 1 | `ICD9` |
| Code | float64 | Code identifier | 0 | 105 | `31.0`, `38.9`, `53.9` |
| Diagnosis_Flag | object | Flag/indicator field | 475 | 2 | `Primary`, `Admitting` |
| Provider | object | Identifier field | 84 | 378 | `Hurtado, Rocio M, MD`, `MGH:00000`, `Christopher, MD, Kenneth B., MD` |
| Clinic | object | Field specific to Dia file | 20 | 205 | `Infectious Disease (23)`, `Microbiology Lab (534)`, `not recorded` |
| Hospital | object | Field specific to Dia file | 0 | 6 | `MGH`, `BWH`, `MEE` |
| Inpatient_Outpatient | object | Visit type | 0 | 5 | `not recorded`, `Outpatient`, `Inpatient` |
| Encounter_number | object | Identifier field | 0 | 886 | `IDX-MGH-3-56492801`, `TSI-MGH-K473023562`, `TSI-BWH-2612200962` |

---

### Dis - Discharge - Discharge summaries and related notes

**File:** `FNR_20240409_091633_Dis.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 3 | 833 | `100004212`, `***This text report has been converted from the report, '1116851658.pdf'. Content may ... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 995 | 1 | `10040032343.0` |
| MRN_Type | object | Field specific to Dis file | 995 | 1 | `MGH` |
| MRN | float64 | Field specific to Dis file | 995 | 1 | `38696.0` |
| Report_Number | object | Identifier field | 995 | 5 | `1116851658`, `1558477093`, `MGHPOE4300059` |
| Report_Date_Time | object | Date field | 995 | 4 | `7/2/2016 12:16:09 PM`, `6/20/2017 4:21:00 PM`, `8/21/2000 4:00:00 AM` |
| Report_Description | object | Field specific to Dis file | 995 | 3 | `Discharge Summary`, `ED Discharge Summary`, `ED Observation Discharge Summary` |
| Report_Status | object | Status of report (final, preliminary, etc.) | 995 | 3 | `F`, `R`, `P` |
| Report_Type | object | Field specific to Dis file | 995 | 14 | `MDISDisch Summ`, `MDISDIS`, `MDISED Obs Disch`, `BDISDIS`, `BDISED Obs Disch`, `BDISDisch Summ`, `FDISDIS`, `WDISDIS`, `NDISDIS`, `WDISDisch Summ`, `FDISDisch Summ`, `SDISDisch Summ`, `SDISDIS`, `NDISDisch Summ` |
| Report_Text | object | Full text of report | 998 | 1 | `Massachusetts General Hospital ` |

---

### Enc - Encounters - Hospital encounters and visits

**File:** `FNR_20240409_091633_Enc.txt`

**Total Columns:** 32

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 9 | `100030788`, `100035740`, `100036484` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 9 | `10040665480`, `10040828823`, `10040859273` |
| MRN_Type | object | Field specific to Enc file | 0 | 1 | `BWH` |
| MRN | int64 | Field specific to Enc file | 0 | 9 | `2566131`, `2750719`, `2776201` |
| Encounter_number | object | Identifier field | 0 | 1000 | `TSI-BWH-9693100029`, `TSI-BWH-9693100030`, `TSI-BWH-9693100031` |
| Encounter_Status | object | Field specific to Enc file | 0 | 4 | `Regular`, `Has Negative Charges Or Cost`, `Cancelled` |
| Hospital | object | Field specific to Enc file | 0 | 1 | `BWH` |
| Inpatient_Outpatient | object | Visit type | 0 | 2 | `Outpatient`, `Inpatient` |
| Service_Line | object | Field specific to Enc file | 561 | 26 | `Primary Care OP`, `Labs OP`, `Radiology OP` |
| Attending_MD | object | Field specific to Enc file | 158 | 283 | `Bienfang, Don C., MD`, `Iacobo, Christina T, MD`, `Workum, Susan Botsford, MD` |
| Admit_Date | object | Hospital admission date | 0 | 796 | `10/29/1996`, `11/05/1996`, `12/13/2006` |
| Discharge_Date | object | Hospital discharge date | 0 | 802 | `10/29/1996`, `11/05/1996`, `12/13/2006` |
| LOS_Days | int64 | Field specific to Enc file | 0 | 7 | `0`, `1`, `2` |
| Clinic_Name | object | Name field | 0 | 48 | `not recorded`, `Neurophysiology (101)`, `BPG At 850 Boylston (100)` |
| Admit_Source | object | Field specific to Enc file | 990 | 4 | `Emergency Room`, `Ambulatory Service`, `Private Physician` |
| Discharge_Disposition | object | Field specific to Enc file | 972 | 4 | `Not Transferred`, `Home Care`, `With Approval` |
| Payor | object | Field specific to Enc file | 14 | 44 | `Harvard Pilgrim Ppo`, `Welfare, Neighborhood Health Plan`, `Neighborhood Health Plan, Welfare` |
| Admitting_Diagnosis | object | Field specific to Enc file | 638 | 135 | `496 - Chronic airway obstruction, not elsewhere classified`, `786.09 - Other dyspnea and respirator... |
| Principal_Diagnosis | object | Field specific to Enc file | 66 | 279 | `114.9 - Coccidioidomycosis, unspecified`, `377.24 - Pseudopapilledema`, `305.1 - Tobacco use disord... |
| Diagnosis_1 | object | Field specific to Enc file | 335 | 260 | `784.0 - Headache`, `285.9 - Anemia, unspecified`, `V70.0 - Routine general medical examination at a... |
| Diagnosis_2 | object | Field specific to Enc file | 705 | 150 | `300.00 - Anxiety state, unspecified`, `733.90 - Disorder of bone and cartilage, unspecified`, `V76.... |
| Diagnosis_3 | object | Field specific to Enc file | 851 | 93 | `V04.81 - Need for prophylactic vaccination and inoculation, Influenza`, `272.0 - Pure hypercholeste... |
| Diagnosis_4 | object | Field specific to Enc file | 923 | 62 | `305.1 - Tobacco use disorder`, `780.79 - Other malaise and fatigue`, `702.19 - Other seborrheic ker... |
| Diagnosis_5 | object | Field specific to Enc file | 955 | 39 | `216.7 - Benign neoplasm of skin of lower limb, including hip`, `493.90 - Asthma, unspecified withou... |
| Diagnosis_6 | object | Field specific to Enc file | 962 | 28 | `530.81 - Esophageal reflux`, `493.20 - Chronic obstructive asthma, without mention of status asthma... |
| Diagnosis_7 | object | Field specific to Enc file | 967 | 23 | `428.0 - Congestive heart failure`, `530.81 - Esophageal reflux`, `250.00 - type II diabetes mellitu... |
| Diagnosis_8 | object | Field specific to Enc file | 974 | 21 | `780.57 - Other and unspecified sleep apnea`, `493.90 - Asthma, unspecified without mention of statu... |
| Diagnosis_9 | object | Field specific to Enc file | 978 | 17 | `135 - Sarcoidosis`, `V12.54 - Personal history of transient ischemic attack (TIA), and cerebral inf... |
| Diagnosis_10 | object | Field specific to Enc file | 981 | 19 | `780.57 - Other and unspecified sleep apnea`, `786.01 - Hyperventilation`, `278.03 - Obesity hypoven... |
| DRG | object | Field specific to Enc file | 972 | 22 | `DRG:277 - Cellulitis Age >17 W CC`, `DRG:076 - Other Resp System O.R. Procedures w cc`, `DRG:049 - ... |
| Patient_Type | float64 | Field specific to Enc file | 1000 | 0 |  |
| Referrer_Discipline | float64 | Field specific to Enc file | 1000 | 0 |  |

---

### Hnp - History & Physical - H&P notes and examinations

**File:** `FNR_20240409_091633_Hnp.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 903 | `100004212`, `***This text report has been converted from the report, '1112311932.pdf'. Content may ... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 995 | 1 | `10040032343.0` |
| MRN_Type | object | Field specific to Hnp file | 995 | 1 | `MGH` |
| MRN | float64 | Field specific to Hnp file | 995 | 1 | `38696.0` |
| Report_Number | float64 | Identifier field | 995 | 5 | `1112311932.0`, `1555739440.0`, `1555724860.0` |
| Report_Date_Time | object | Date field | 995 | 5 | `6/30/2016 11:42:12 AM`, `6/18/2017 7:49:20 PM`, `6/18/2017 6:54:53 PM` |
| Report_Description | object | Field specific to Hnp file | 995 | 13 | `MHPHP`, `BHPHP`, `NHPHP`, `VHPHP`, `WHPHP`, `MHPHP(View Onl`, `FHPHP`, `BHPHP(View Onl`, `SHPHP`, `WHPHP(View Onl`, `FHPHP(View Onl`, `VHPHP(View Onl`, `NHPHP(View Onl` |
| Report_Status | object | Status of report (final, preliminary, etc.) | 995 | 2 | `F`, `R` |
| Report_Type | object | Field specific to Hnp file | 995 | 13 | `MHPHP`, `BHPHP`, `NHPHP`, `VHPHP`, `WHPHP`, `MHPHP(View Onl`, `FHPHP`, `BHPHP(View Onl`, `SHPHP`, `WHPHP(View Onl`, `FHPHP(View Onl`, `VHPHP(View Onl`, `NHPHP(View Onl` |
| Report_Text | float64 | Full text of report | 1000 | 0 |  |

---

### Lab - Laboratory - Lab test results and values

**File:** `FNR_20240409_091633_Lab.txt`

**Total Columns:** 22

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 3 | `100118964`, `100071500`, `100094988` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 3 | `10045539268`, `10042818004`, `10025274159` |
| MRN_Type | object | Field specific to Lab file | 0 | 1 | `BWH` |
| MRN | int64 | Field specific to Lab file | 0 | 3 | `27376771`, `7504459`, `4259891` |
| Seq_Date_Time | object | Date field | 0 | 272 | `11/23/2020 07:45`, `11/22/2020 22:20`, `06/04/1990 16:42` |
| Group_Id | object | Identifier field | 50 | 30 | `TROPTHS`, `ALB`, `CA` |
| Loinc_Code | object | Code identifier | 0 | 31 | `67151-1`, `1751-7`, `17861-6` |
| Test_Id | object | Identifier field | 0 | 47 | `TNTHS`, `ALB`, `CA` |
| Test_Description | object | Field specific to Lab file | 0 | 49 | `TROPONIN-T, HS GEN5 (Test:bctnths)`, `Albumin (Test:bcalb)`, `Calcium (Test:bcca)` |
| Result | object | Result/value field | 0 | 400 | `116.00000`, `4.10000`, `10.00000` |
| Result_Text | object | Result/value field | 18 | 23 | `Test Performed By:\.br\BWH Clinical Laboratories, Director: Milenko Tanasijevic, MD\.br\75 Francis ... |
| Abnormal_Flag | object | Indicates if result is abnormal | 422 | 2 | `High`, `Low` |
| Reference_Units | object | Field specific to Lab file | 4 | 10 | `ng/L`, `g/dL`, `mg/dL` |
| Reference_Range | object | Normal reference range | 4 | 49 | `0-9`, `3.5-5.2`, `8.8-10.7` |
| Toxic_Range | float64 | Field specific to Lab file | 1000 | 0 |  |
| Specimen_Type | float64 | Field specific to Lab file | 1000 | 0 |  |
| Specimen_Text | float64 | Text/note field | 1000 | 0 |  |
| Correction_Flag | float64 | Flag/indicator field | 1000 | 0 |  |
| Test_Status | float64 | Field specific to Lab file | 1000 | 0 |  |
| Ordering_Doc_Name | object | Name field | 18 | 16 | `Prange, Hannah Ryann`, `O'Meara, Tess Ann, MD`, `Shekar, Prem S, MD` |
| Accession | object | Field specific to Lab file | 0 | 317 | `M8173851_335L738910`, `X3848722_335L729800`, `83489` |
| Source | object | Field specific to Lab file | 0 | 1 | `RPDR` |

---

### Let - Letters - Clinical letters and correspondence

**File:** `FNR_20240409_091633_Let.txt`

⚠️ **Error reading file:** Error tokenizing data. C error: Buffer overflow caught - possible malformed input file.


### Lno - Lab Notes - Laboratory notes and comments

**File:** `FNR_20240409_091633_Lno.txt`

**Total Columns:** 13

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 830 | `100003300`, `Subject: [Follow Up Monitoring] Reason for Hospital Admission: Ms Spurill is a 68 year... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 980 | 1 | `10040025388.0` |
| MRN_Type | float64 | Field specific to Lno file | 1000 | 0 |  |
| MRN | float64 | Field specific to Lno file | 1000 | 0 |  |
| LMRNote_Date | object | Date field | 980 | 19 | `4/19/2010 12:00:00 AM`, `3/9/2011 12:00:00 AM`, `6/6/2011 12:00:00 AM` |
| Record_Id | float64 | Identifier field | 980 | 20 | `51001763.0`, `63546357.0`, `67202908.0` |
| Status | object | Field specific to Lno file | 980 | 1 | `F` |
| Author | object | Field specific to Lno file | 981 | 12 | `Lew, MD, Kathryn Malakorn, MD`, `Gramajo, Ana Lia`, `Celi, Ann C., MD,MPH` |
| COD | object | Field specific to Lno file | 980 | 13 | `B002333244`, `B001867877`, `B000187705` |
| Institution | object | Field specific to Lno file | 980 | 1 | `BWH` |
| Author_MRN | float64 | Field specific to Lno file | 1000 | 0 |  |
| Subject | object | Field specific to Lno file | 980 | 13 | `Follow Up Monitoring`, `Result Manager Letter`, `Patient Note` |
| Comments | object | Field specific to Lno file | 994 | 3 | `Subject: [Free Form Text]`, `Subject: [Influenza Vaccine]`, `Subject: [Patient Note]` |

---

### Log - Log - System log entries

**File:** `FNR_20240409_091633_Log.txt`

**Total Columns:** 1

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| Environment Settings:--------------------------------------------------------------------------- | object | Field specific to Log file | 0 | 425 | `	Machine	 name:		MGBAPP2364`, `	Operating system:	Microsoft Windows NT 6.2.9200.0`, `	Login	 user:	... |

---

### Med - Medications - Medication prescriptions and administrations

**File:** `FNR_20240409_091633_Med.txt`

**Total Columns:** 16

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 9 | `100030788`, `100035740`, `100036484` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 9 | `10040665480`, `10040828823`, `10040859273` |
| MRN_Type | object | Field specific to Med file | 0 | 1 | `BWH` |
| MRN | int64 | Field specific to Med file | 0 | 9 | `2566131`, `2750719`, `2776201` |
| Medication_Date | object | Date field | 0 | 215 | `1/8/2009`, `12/9/2010`, `10/24/2014` |
| Medication_Date_Detail | float64 | Date field | 1000 | 0 |  |
| Medication | object | Field specific to Med file | 0 | 437 | `Influenza tvs 08 -09 vaccine/pf 45 mcg disp syrin.5ml syringe`, `Influenza virus vaccine, split vir... |
| Code_Type | object | Type of code (ICD-9, ICD-10, CPT, etc.) | 0 | 3 | `BWH_CC`, `CPT`, `HCPCS` |
| Code | object | Code identifier | 0 | 460 | `945826`, `90656`, `941354` |
| Quantity | float64 | Field specific to Med file | 0 | 33 | `1.0`, `2.0`, `11.0` |
| Provider | object | Identifier field | 12 | 109 | `Iacobo, Christina T, MD`, `Englert, Joshua A., MD`, `Czuczman,  , Gregory J, MD` |
| Clinic | object | Field specific to Med file | 0 | 18 | `BPG At 850 Boylston (100)`, `BWH Pulmonary Medicine (850)`, `BWH Brigham & Womens Physician Grp (92... |
| Hospital | object | Field specific to Med file | 0 | 1 | `BWH` |
| Inpatient_Outpatient | object | Visit type | 0 | 2 | `Outpatient`, `Inpatient` |
| Encounter_number | object | Identifier field | 0 | 135 | `TSI-BWH-9693100035`, `TSI-BWH-9693100053`, `TSI-BWH-9693100092` |
| Additional_Info | object | Field specific to Med file | 0 | 1 | ` ` |

---

### Mic - Microbiology - Microbiology culture results

**File:** `FNR_20240409_091633_Mic.txt`

**Total Columns:** 16

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 243 | `100004212`, `_______________________________________________________________________________`, `Spe... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 936 | 2 | `10040032343.0`, `10040029737.0` |
| MRN_Type | object | Field specific to Mic file | 936 | 2 | `MGH`, `BWH` |
| MRN | float64 | Field specific to Mic file | 936 | 2 | `38696.0`, `667360.0` |
| Microbiology_Number | object | Identifier field | 936 | 47 | `M6278458_322L387972`, `M6278459_322L387973`, `M6278469_322L387983` |
| Microbiology_Date_Time | object | Date field | 936 | 45 | `10/15/2019 00:21:00`, `10/15/2019 00:12:00`, `09/16/2019 15:14:00` |
| Specimen_Type | object | Field specific to Mic file | 936 | 901 | `BLOOD`, `URINE`, `NASAL`, `NASOPHARYNGEAL SWAB`, `MOUTH`, `TONGUE`, `GASTRIC BIOPSY`, `STOOL`, `CATHETERIZED URINE`, `VAGINA`, `BLOOD/SERUM FOR VIROLOGY`, `NAIL`, `ABDOMINAL FLUID`, `SPUTUM`, `INDUCED SPUTUM (BWH) FOR AFB CULTURE AND STAIN`, `SPUTUM (INDUCED)`, `CERVIX FOR CHLAMYDIA/N. GONORRHOEAE BY GEN PROBE`, `URINE (RECEIVED ON PLATED MEDIA)`, `FLUID`, `CYST` (20 of 901 unique values) |
| Specimen_Comments | object | Field specific to Mic file | 959 | 20 | `BLOOD SET 1`, `BLOOD SET 2`, `URINE ` |
| Test_Name | object | Name field | 993 | 597 | `ADENOVIRUS ANTIGEN`, `METAPNEUMOVIRUS Ag`, `PARAINFLUENZA 1-3 AG`, `C. Difficile Toxin`, `HP HELICOBACTER PYLORI IGG EIA`, `GRAM STAIN`, `C. DIFFICILE ANTIGEN/TOXIN ASSAY`, `AEROBIC CULTURE`, `Beta-D-Glucan (1-3)`, `OVA & PARASITES`, `VANCOMYCIN-RESISTANT ENTEROCOCCUS SCREEN (ADULTS)`, `MRSA CULTURE (ADULTS)`, `CMV IgM (EIA)`, `EPSTEIN-BARR EARLY AG IGG`, `Gram Stain`, `Acid Fast Smear`, `Fungal Wet Prep`, `FLU A&B SCREEN`, `RESP CULTURE`, `STREP PNEUMO ANTIGEN - URINE` (20 of 597 unique values) |
| Test_Code | object | Code identifier | 993 | 6 | `SQ-ADENO`, `SQ-METAV`, `SQ-PCAD` |
| Test_Comments | object | Field specific to Mic file | 993 | 6 | `NEGATIVE `, ` NEGATIVE`, ` NEGATIVE for Clostridium difficile toxin.` |
| Test_Status | object | Field specific to Mic file | 936 | 5 | `F`, `P`, `C`, `I`, `O` |
| Organism_Name | object | Name field | 944 | 3448 | `NO GROWTH 7 DAYS`, `NO GROWTH`, `ENTEROCOCCUS`, `ENTERIC GRAM NEGATIVE RODS`, `STAPHYLOCOCCUS SPECIES`, `MIXED BACTERIA`, `NEGATIVE FOR MRSA`, `MIXED BACTERIA WITH 2 POTENTIAL PATHOGENS IN RARE AMOUNT`, `NO VIRUS ISOLATED AFTER 7 DAYS`, `NO VIRUS ISOLATED AFTER 9 DAYS`, `NEGATIVE at 24 hour incubation`, `MIXED BACTERIA WITH 1 POTENTIAL PATHOGEN IN RARE AMOUNT`, `NEGATIVE FOR ENTERIC PATHOGENS`, `MIXED FLORA (3 OR MORE COLONY TYPES)`, `NO SALMONELLA, SHIGELLA, AEROMONAS OR PLESIOMONAS ISOLATED`, `NO CAMPYLOBACTER ISOLATED`, `NO ESCHERICHIA COLI 0157:H7 ISOLATED`, `NEGATIVE FOR SHIGA TOXIN 1 AND 2`, `VAGINAL FLORA`, `MOLD. SENT TO MYCOLOGY LABORATORY FOR IDENTIFICATION` (20 of 3448 unique values) |
| Organism_Code | object | Code identifier | 944 | 33 | `NG7`, `NG`, `ENTC` |
| Organism_Comment | object | Field specific to Mic file | 984 | 11 | `Rare (100 to <1000 CFU/ml) MIXED BACTERIA`, `Few (1000 to <10,000 CFU/ml) MIXED BACTERIA`, `NO VIRU... |
| Organism_Text | object | Text/note field | 936 | 1 | `Microbiology Report` |

---

### Mrn - Medical Record Numbers - Patient MRN mappings

**File:** `FNR_20240409_091633_Mrn.txt`

**Total Columns:** 15

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| IncomingId | int64 | Identifier field | 0 | 1000 | `100004212`, `100003884`, `100004796` |
| IncomingSite | object | Field specific to Mrn file | 0 | 1 | `EMP` |
| Status | object | Field specific to Mrn file | 0 | 1 | `A` |
| Enterprise_Master_Patient_Index | int64 | Field specific to Mrn file | 0 | 1000 | `100004212`, `100003884`, `100004796` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 1000 | `10040032343`, `10040029737`, `10033842823` |
| MGH_MRN | float64 | Massachusetts General Hospital MRN | 41 | 959 | `38696.0`, `4788338.0`, `326078.0` |
| BWH_MRN | float64 | Brigham and Women's Hospital MRN | 121 | 879 | `9556150.0`, `667360.0`, `782771.0` |
| FH_MRN | float64 | Field specific to Mrn file | 490 | 510 | `662134.0`, `1158621.0`, `1224192.0` |
| SRH_MRN | float64 | Field specific to Mrn file | 617 | 383 | `100107789.0`, `100176131.0`, `422066.0` |
| NWH_MRN | float64 | Field specific to Mrn file | 320 | 680 | `11572411.0`, `10354015.0`, `209178.0` |
| NSMC_MRN | float64 | Field specific to Mrn file | 667 | 333 | `364545.0`, `355366.0`, `10969716.0` |
| MCL_MRN | float64 | Field specific to Mrn file | 913 | 87 | `280928.0`, `245370.0`, `111812.0` |
| MEE_MRN | float64 | Field specific to Mrn file | 293 | 707 | `7327588.0`, `7798371.0`, `9009568.0` |
| DFC_MRN | float64 | Field specific to Mrn file | 436 | 564 | `931291.0`, `606305.0`, `541806.0` |
| WDH_MRN | float64 | Field specific to Mrn file | 952 | 48 | `809388.0`, `829486.0`, `813827.0` |

---

### Opn - Operative Notes - Surgical and procedure notes

**File:** `FNR_20240409_091633_Opn.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 838 | `100004212`, `NAME:  BENTIVEGNA, PHILIP J         UNIT NO:  003-86-96`, `DATE: 06/18/1999        FLO... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 984 | 2 | `10040032343.0`, `10040029737.0` |
| MRN_Type | object | Field specific to Opn file | 984 | 3 | `MGH`, `BWH`, `FH` |
| MRN | float64 | Field specific to Opn file | 984 | 3 | `38696.0`, `667360.0`, `662134.0` |
| Report_Number | object | Identifier field | 984 | 16 | `256451`, `353121`, `427040` |
| Report_Date_Time | object | Date field | 984 | 16 | `6/22/1999 12:00:00 AM`, `8/23/2000 12:00:00 AM`, `6/21/2001 12:00:00 AM` |
| Report_Description | object | Field specific to Opn file | 984 | 18 | `MOPNOPN`, `BOPNOPN`, `FOPNOPN`, `BOPNOP NOTE`, `VOPNOPN`, `MOPNBRIEF OP NOT`, `MOPNOP NOTE`, `BOPNBRIEF OP NOT`, `VOPNBRIEF OP NOT`, `VOPNOP NOTE`, `WOPNOPN`, `WOPNBRIEF OP NOT`, `WOPNOP NOTE`, `FOPNBRIEF OP NOT`, `NOPNOPN`, `NOPNBRIEF OP NOT`, `FOPNOP NOTE`, `NOPNOP NOTE` |
| Report_Status | object | Status of report (final, preliminary, etc.) | 984 | 3 | `F`, `P`, `R` |
| Report_Type | object | Field specific to Opn file | 984 | 18 | `MOPNOPN`, `BOPNOPN`, `FOPNOPN`, `BOPNOP NOTE`, `VOPNOPN`, `MOPNBRIEF OP NOT`, `MOPNOP NOTE`, `BOPNBRIEF OP NOT`, `VOPNBRIEF OP NOT`, `VOPNOP NOTE`, `WOPNOPN`, `WOPNBRIEF OP NOT`, `WOPNOP NOTE`, `FOPNBRIEF OP NOT`, `NOPNOPN`, `NOPNBRIEF OP NOT`, `FOPNOP NOTE`, `NOPNOP NOTE` |
| Report_Text | object | Full text of report | 986 | 8 | `OPERATIVE REPORT`, `Massachusetts General Hospital`, `Report Status:  Unsigned` |

---

### Pat - Pathology - Pathology reports and results

**File:** `FNR_20240409_091633_Pat.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 795 | `100003884`, `Type:  Surgical Pathology`, `Pathology Report:  BS-22-X30373` |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 985 | 6 | `10040029737.0`, `10040017377.0`, `10026394246.0` |
| MRN_Type | object | Field specific to Pat file | 985 | 3 | `BWH`, `MGH`, `FH` |
| MRN | float64 | Field specific to Pat file | 985 | 6 | `667360.0`, `8054.0`, `4232120.0` |
| Report_Number | object | Identifier field | 985 | 15 | `BS22X30373`, `BS22M62783`, `S24-7272` |
| Report_Date_Time | object | Date field | 985 | 15 | `5/23/2022 1:19:00 PM`, `10/17/2022 3:54:00 PM`, `1/31/2024 1:02:00 PM` |
| Report_Description | object | Field specific to Pat file | 985 | 47 | `BPATS`, `ANATOMIC PATHOLOGY`, `PROTEIN ELECTROPHORESIS`, `FPATS`, `AUTOPSY, GENERAL`, `MPATS`, `NON-GYN CYTOLOGY`, `MOLECULAR DIAGNOSTICS`, `MPATC`, `INTERPRETIVE LAB REPORT`, `FLOW CYTOMETRY`, `RAPID HEME PANEL`, `CYTOGENETICS`, `BPATC`, `PAP SMEAR`, `PROFILE SOMATIC ONCOPANEL`, `SPECIAL COAGULATION`, `WPATS`, `DERMATOPATHOLOGY`, `OUTSIDE PATHOLOGY REVIEW` (20 of 47 unique values) |
| Report_Status | object | Status of report (final, preliminary, etc.) | 985 | 24 | `F`, `R`, `P`, `Final`, `Updated`, `Signed`, `Corrected`, `Amend/Addenda`, `Revised`, `Completed`, `Deleted`, `Cancelled`, `Preliminary`, `In Process`, `Pending`, `Not Verified`, `Amended`, `In Revision`, `Hold`, `Incomplete` (20 of 24 unique values) |
| Report_Type | object | Field specific to Pat file | 985 | 41 | `BPATS`, `EPAT66997`, `EPAT444417`, `FPATS`, `EPAT444428`, `MPATS`, `EPAT444396`, `EPAT444402`, `MPATC`, `EPAT444433`, `EPAT444427`, `EPAT444413`, `EPAT444400`, `BPATC`, `EPAT444394`, `EPAT559797`, `EPAT444415`, `WPATS`, `EPAT444406`, `EPAT444388` (20 of 41 unique values) |
| Report_Text | object | Full text of report | 985 | 15 | `Accession Number:  BS22X30373                     Report Status:  Final`, `Accession Number:  BWHPo... |

---

### Phy - Physicians - Provider information

**File:** `FNR_20240409_091633_Phy.txt`

**Total Columns:** 15

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 2 | `100004212`, `100003884` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 2 | `10040032343`, `10040029737` |
| MRN_Type | object | Field specific to Phy file | 500 | 4 | `MGH`, `BWH`, `FH` |
| MRN | float64 | Field specific to Phy file | 500 | 4 | `38696.0`, `667360.0`, `662134.0` |
| Date | object | Date field | 0 | 184 | `10/14/2016`, `11/6/2017`, `10/8/2018` |
| Concept_Name | object | Name field | 0 | 41 | `Flu-High Dose`, `BMI`, `Blood Pressure-Epic` |
| Code_Type | object | Type of code (ICD-9, ICD-10, CPT, etc.) | 70 | 2 | `EPIC`, `LMR` |
| Code | object | Code identifier | 70 | 46 | `76`, `BMI`, `BP` |
| Result | object | Result/value field | 128 | 326 | `25.31`, `22.8`, `23.13` |
| Units | object | Field specific to Phy file | 653 | 10 | `millimeter of mercury`, `inch`, `beats/minute` |
| Provider | object | Identifier field | 72 | 68 | `Wheeler, Amy Ellen, MD`, `Vickery, Amanda Jayne, CNP`, `Mchugh, Michele, CNP` |
| Clinic | object | Field specific to Phy file | 0 | 30 | `MGH ADLT MED RHC (10020050002)`, `MGH PULM ASSC COX2 (10020010197)`, `MGH ADLT MED CTHC1 (100200400... |
| Hospital | object | Field specific to Phy file | 48 | 4 | `MGH`, `BWH`, `FH` |
| Inpatient_Outpatient | object | Visit type | 0 | 3 | `Outpatient`, `Outpatient-Emergency`, `Inpatient` |
| Encounter_number | object | Identifier field | 0 | 391 | `EPIC-3114970472`, `EPIC-3163126330`, `EPIC-3200352367` |

---

### Prc - Procedures - CPT procedure codes

**File:** `FNR_20240409_091633_Prc.txt`

**Total Columns:** 15

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 95 | `100121396`, `100093092`, `100124276` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 95 | `10045679502`, `10021071377`, `10045841656` |
| MRN_Type | object | Field specific to Prc file | 0 | 6 | `BWH`, `MGH`, `MEE` |
| MRN | int64 | Field specific to Prc file | 0 | 137 | `7315385`, `1089707`, `3108902` |
| Date | object | Date field | 0 | 797 | `7/13/2022`, `7/15/2010`, `10/16/2017` |
| Procedure_Name | object | Procedure description | 0 | 257 | `Anesthesia for access to central venous circulation`, `Anesthesia for upper gastrointestinal endosc... |
| Code_Type | object | Type of code (ICD-9, ICD-10, CPT, etc.) | 0 | 1 | `CPT` |
| Code | object | Code identifier | 0 | 257 | `00532`, `00740`, `00840` |
| Procedure_Flag | float64 | Flag/indicator field | 1000 | 0 |  |
| Quantity | float64 | Field specific to Prc file | 8 | 32 | `11.0`, `1.0`, `16.0` |
| Provider | object | Identifier field | 14 | 552 | `Kotova, Faina, MD`, `White, Rodger Musser, MD`, `Nanji, Karen Caputo, MD, MPH, ScD` |
| Clinic | object | Field specific to Prc file | 19 | 243 | `BWH IMG IR ANGIO (10030010286)`, `DACC (10)`, `MGH PERIOPERATIVE DEPT (10020011358)` |
| Hospital | object | Field specific to Prc file | 0 | 6 | `BWH`, `MGH`, `MEE` |
| Inpatient_Outpatient | object | Visit type | 0 | 4 | `Outpatient`, `Inpatient`, `Outpatient-Emergency` |
| Encounter_number | object | Identifier field | 0 | 932 | `EPIC-PB-BWH-6197848503`, `IDX-MGH-3-49560141`, `EPIC-PB-MGH-6065288363` |

---

### Prg - Progress Notes - Clinical progress notes

**File:** `FNR_20240409_091633_Prg.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 782 | `100004212`, `***This text report has been converted from the report, '970630973.pdf'. Content may n... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 989 | 1 | `10040032343.0` |
| MRN_Type | object | Field specific to Prg file | 989 | 1 | `MGH` |
| MRN | float64 | Field specific to Prg file | 989 | 1 | `38696.0` |
| Report_Number | float64 | Identifier field | 989 | 11 | `970630973.0`, `1112435159.0`, `1113941780.0` |
| Report_Date_Time | object | Date field | 989 | 11 | `4/10/2016 4:20:56 PM`, `6/29/2016 6:17:15 PM`, `6/30/2016 2:50:32 PM` |
| Report_Description | object | Field specific to Prg file | 989 | 25 | `MPRGPROGRESS`, `MPRGNursing`, `MPRGEvent`, `MPRGED Obs Prog`, `BPRGPROGRESS`, `FPRGPROGRESS`, `NPRGPROGRESS`, `BPRGEvent`, `WPRGPROGRESS`, `BPRGNursing`, `MPRGPeriop Nursi`, `BPRGED Obs Prog`, `WPRGNursing`, `FPRGEvent`, `FPRGNursing`, `BPRGPeriop Nursi`, `SPRGPROGRESS`, `SPRGEvent`, `WPRGPeriop Nursi`, `FPRGED Obs Prog` (20 of 25 unique values) |
| Report_Status | object | Status of report (final, preliminary, etc.) | 989 | 2 | `F`, `R` |
| Report_Type | object | Field specific to Prg file | 989 | 25 | `MPRGPROGRESS`, `MPRGNursing`, `MPRGEvent`, `MPRGED Obs Prog`, `BPRGPROGRESS`, `FPRGPROGRESS`, `NPRGPROGRESS`, `BPRGEvent`, `WPRGPROGRESS`, `BPRGNursing`, `MPRGPeriop Nursi`, `BPRGED Obs Prog`, `WPRGNursing`, `FPRGEvent`, `FPRGNursing`, `BPRGPeriop Nursi`, `SPRGPROGRESS`, `SPRGEvent`, `WPRGPeriop Nursi`, `FPRGED Obs Prog` (20 of 25 unique values) |
| Report_Text | float64 | Full text of report | 1000 | 0 |  |

---

### Pul - Pulmonary - Pulmonary function tests

**File:** `FNR_20240409_091633_Pul.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 4 | 635 | `100004212`, `Name: BENTIVEGNA, PHILIP J.   Unit #: 0038696          Date: 5/23/2019 12:03pm`, `Loca... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 983 | 4 | `10040032343.0`, `10040029737.0`, `10033842823.0` |
| MRN_Type | object | Field specific to Pul file | 983 | 3 | `MGH`, `BWH`, `FH` |
| MRN | float64 | Field specific to Pul file | 983 | 5 | `38696.0`, `667360.0`, `1158621.0` |
| Report_Number | object | Identifier field | 983 | 17 | `MGHCOMPASPUL{1A352350-E443-47AD-9617-C8291817C657}FILE1`, `MGHCOMPASPUL{1A352350-E443-47AD-9617-C82... |
| Report_Date_Time | object | Date field | 983 | 10 | `5/31/2019 4:07:43 PM`, `3/3/2019 4:46:21 PM`, `10/24/2011 4:00:00 AM` |
| Report_Description | object | Field specific to Pul file | 983 | 94 | `PFT/ABG Study`, `PULMONARY FUNCTION TEST`, `PULMONARY FUNCTION REPORT`, `PFT/ABG STUDIES`, `PFT(pre & post) Volume DLCO`, `PFT(pre & post)`, `PFT(pre & post) Sat`, `PFT`, `PFT(pre & post) Volume, Sat`, `Test Data Pending`, `PFT(pre & post) Volume DLCO Sat`, `PFT Volume DLCO Sat`, `PFT Volume`, `PFT Volume DLCO`, `PFT(pre & post) Volume Sat`, `PFT DLCO`, `PFT Sat`, `PFT(pre & post) Volume DLCO Muscle Forces`, `METHACHOLINE` (20 of 94 unique values) |
| Report_Status | object | Status of report (final, preliminary, etc.) | 983 | 3 | `LA*Final`, `ZA*Amended`, `PA*Preliminary` |
| Report_Type | object | Field specific to Pul file | 983 | 1 | `PUL*Pulmonary` |
| Report_Text | object | Full text of report | 990 | 8 | `The Pulmonary document is available for viewing and printing in the EHR Viewers/Results.  If this i... |

---

### Rad - Radiology - Radiology reports and imaging results

**File:** `FNR_20240409_091633_Rad.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 0 | 735 | `100004212`, `Left shoulder: three views.`, `Comparison: none.` |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 960 | 2 | `10040032343.0`, `10040029737.0` |
| MRN_Type | object | Field specific to Rad file | 960 | 2 | `MGH`, `BWH` |
| MRN | float64 | Field specific to Rad file | 960 | 2 | `38696.0`, `667360.0` |
| Report_Number | object | Identifier field | 960 | 40 | `8918887`, `9131721`, `9882146` |
| Report_Date_Time | object | Date field | 960 | 35 | `5/9/2005 9:50:00 AM`, `8/24/2005 9:01:00 AM`, `8/25/2006 6:24:00 PM` |
| Report_Description | object | Field specific to Rad file | 960 | 3751 | `Shoulder 2 Views (Test:XRSHD)`, `PA and Lateral Chest (Test:XRCH2)`, `UltrasdVasc,AortalVC/iliacGrft (Test:2630CH)`, `Asp &/or Inj Maj Jt w Fluoro (Test:RFJT1)`, `CT Sholder W/O Con (Test:CTSHDWO)`, `Shoulder Arthrography (Test:RFSHD)`, `TCD-Transcranial Doppler - Ltd (Test:USVA24)`, `MRI Lumbar Spn Neuro W/O Con (Test:MRILSNWO)`, `Outside - Plain Film Lower Extremity (Test:XRLOWEXTLI)`, `XR CHEST (Test:XR.TH.CHEST)`, `CT HEAD (Test:CT.NE.HEAD)`, `Lumbo-sacral Spine,AP &Lateral (Test:XRLS1)`, `US LOWER EXTREMITY VEINS DUPLEX (BILATERAL)`, `CT CHEST PULMONARY ANGIOGRAM (Test:CT.TH.CHESTPE)`, `Outside - Plain Film Spine (Test:XRSPINELI)`, `CT CHEST (Test:CT.TH.CHEST)`, `CT CERVICAL SPINE (Test:CT.XS.CSPINE)`, `XR CHEST PORTABLE (Test:XR.TH.CXRPOR)`, `FL (SPEECH) VIDEO SWALLOW STUDY`, `CT ABDOMEN/PELVIS` (20 of 3751 unique values) |
| Report_Status | object | Status of report (final, preliminary, etc.) | 960 | 2 | `F`, `R` |
| Report_Type | object | Field specific to Rad file | 960 | 4640 | `MRRADXRSHD`, `MRRADXRCH2`, `MRRAD2630CH`, `MRRADRFJT1`, `MRRADCTSHDWO`, `MRRADRFSHD`, `MRRADUSVA24`, `MRRADMRILSNWO`, `MRRADXRLOWEXTLI`, `MRRADXR.TH.CHEST`, `MRRADCT.NE.HEAD`, `MRRADXRLS1`, `MRRADUS.VA.VVELEX/B`, `MRRADCT.TH.CHESTPE`, `MRRADXRSPINELI`, `MRRADCT.TH.CHEST`, `MRRADCT.XS.CSPINE`, `MRRADXR.TH.CXRPOR`, `MRRADFL.TH.VISWAL`, `MRRADCT.AB.ABDPEL` (20 of 4640 unique values) |
| Report_Text | object | Full text of report | 960 | 31 | `PAIN R/O TENDONITIS`, `FOLLOW UP NODULE`, `Cough 75 y.o. male with 3 days cough, RA O2 sat 96% -- r... |

---

### Rdt - Radiology Details - Additional radiology information

**File:** `FNR_20240409_091633_Rdt.txt`

**Total Columns:** 14

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 17 | `100004212`, `100003884`, `100004796` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 17 | `10040032343`, `10040029737`, `10033842823` |
| MRN_Type | object | Field specific to Rdt file | 0 | 2 | `MGH`, `BWH` |
| MRN | int64 | Field specific to Rdt file | 0 | 20 | `38696`, `667360`, `782771` |
| Date | object | Date field | 0 | 654 | `5/9/2005`, `8/23/2005`, `8/25/2006` |
| Mode | object | Field specific to Rdt file | 0 | 15 | `XR`, `US`, `XR.FLUOR` |
| Group | object | Field specific to Rdt file | 0 | 28 | `UEXTR`, `CHEST`, `LEXTR` |
| Test_Code | object | Code identifier | 0 | 332 | `MRRADXRSHD`, `MRRADXRCH2`, `MRRAD2630CH` |
| Test_Description | object | Field specific to Rdt file | 0 | 317 | `Shoulder 2 Views (Test:XRSHD)`, `PA and Lateral Chest (Test:XRCH2)`, `UltrasdVasc,AortalVC/iliacGrf... |
| Accession_Number | object | Identifier field | 1 | 999 | `8918887`, `9131721`, `9882146` |
| Provider | object | Identifier field | 401 | 246 | `Wheeler, Amy Ellen, MD`, `Warner, Jon J. P., MD`, `Kelly,  , Erin Elizabeth, PA` |
| Clinic | object | Field specific to Rdt file | 220 | 14 | `not recorded`, `Chelsea Health Center`, `Revere Health Center` |
| Hospital | object | Field specific to Rdt file | 0 | 2 | `MGH`, `BWH` |
| Inpatient_Outpatient | object | Visit type | 0 | 5 | `not recorded`, `Outpatient`, `Outpatient-Emergency` |

---

### Rfv - Reason for Visit - Chief complaints and visit reasons

**File:** `FNR_20240409_091633_Rfv.txt`

**Total Columns:** 13

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 7 | `100004212`, `100003884`, `100004796` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 7 | `10040032343`, `10040029737`, `10033842823` |
| MRN_Type | float64 | Field specific to Rfv file | 1000 | 0 |  |
| MRN | float64 | Field specific to Rfv file | 1000 | 0 |  |
| Start_date | object | Date field | 0 | 745 | `10/15/2019 5:13:00 AM`, `5/23/2018 3:10:00 PM`, `2/12/2019 1:14:00 PM` |
| End_date | object | Date field | 0 | 745 | `10/18/2019 5:43:00 PM`, `5/24/2018 3:51:00 PM`, `2/13/2019 1:45:00 PM` |
| Provider | object | Identifier field | 63 | 238 | `Saukkonen, Kai, MD`, `Ota, H Gregory, MD`, `Sweetnam, Sandra Margaret, MD` |
| Hospital | object | Field specific to Rfv file | 0 | 7 | `MGH`, `BWH`, `MEE` |
| Clinic | object | Field specific to Rfv file | 0 | 97 | `MGH BIGELOW 11 MED (10020010610)`, `MGH EMERGENCY (10020010608)`, `MGH ELLISON 10 STP DWN (10020010... |
| Chief_complaint | object | Field specific to Rfv file | 0 | 132 | `Other`, `Shortness of Breath`, `Altered Mental Status` |
| Concept_id | object | Identifier field | 0 | 133 | `ERFV:0`, `ERFV:100001`, `ERFV:160032` |
| Comments | object | Field specific to Rfv file | 872 | 112 | `L ear wax. Denies hearing problems.`, `pulmonary`, `lantus` |
| Encounter_number | object | Identifier field | 0 | 947 | `EPIC-3274284152`, `EPIC-3202525412`, `EPIC-3237567273` |

---

### Trn - Transfusions - Blood product transfusions

**File:** `FNR_20240409_091633_Trn.txt`

**Total Columns:** 11

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | int64 | Enterprise Master Patient Identifier - Unique patient ID | 0 | 22 | `100004212`, `100003884`, `100004796` |
| EPIC_PMRN | int64 | EPIC Patient Medical Record Number | 0 | 22 | `10040032343`, `10040029737`, `10033842823` |
| MRN_Type | object | Field specific to Trn file | 0 | 6 | `MGH`, `BWH`, `DFC` |
| MRN | int64 | Field specific to Trn file | 0 | 28 | `38696`, `667360`, `782771` |
| Transaction_Date_Time | object | Date field | 0 | 365 | `10/15/2019 05:47`, `03/05/2019 00:33`, `09/18/2018 12:14` |
| Test_Description | object | Field specific to Trn file | 55 | 81 | `ABO`, `RH`, `Antibody Screen` |
| Abnormal_Flag | object | Indicates if result is abnormal | 995 | 1 | `*` |
| Status_Flag | object | Flag/indicator field | 166 | 6 | `F`, `V`, `I` |
| Comments | object | Field specific to Trn file | 513 | 41 | `; Test Performed By: MGH Department of Pathology, Director: Kent Lewandrowski, M.D. 55 Fruit Street... |
| Results | object | Result/value field | 55 | 383 | `A`, `Negative`, `W061719010247-9` |
| Accession | object | Field specific to Trn file | 0 | 458 | `596154730 L3220488873`, `596160245 L3220488873`, `511888402 L3150107508` |

---

### Vis - Vital Signs - Vital sign measurements

**File:** `FNR_20240409_091633_Vis.txt`

**Total Columns:** 10

| Column Name | Data Type | Description | Null Count | Unique Values | Sample Values |
|-------------|-----------|-------------|------------|---------------|---------------|
| EMPI | object | Enterprise Master Patient Identifier - Unique patient ID | 2 | 822 | `100004212`, `***This text report has been converted from the report, 'VISIT3100486846.pdf'. Content... |
| EPIC_PMRN | float64 | EPIC Patient Medical Record Number | 996 | 1 | `10040032343.0` |
| MRN_Type | object | Field specific to Vis file | 996 | 1 | `MGH` |
| MRN | float64 | Field specific to Vis file | 996 | 1 | `38696.0` |
| Report_Number | object | Identifier field | 996 | 4 | `VISIT3100486846`, `1110534024`, `1110484091` |
| Report_Date_Time | object | Date field | 996 | 4 | `4/8/2016 1:00:00 PM`, `6/29/2016 1:02:26 AM`, `6/30/2016 10:57:49 PM` |
| Report_Description | object | Field specific to Vis file | 996 | 3 | `Office Visit`, `ED Progress/Update Note`, `ED Provider Note` |
| Report_Status | object | Status of report (final, preliminary, etc.) | 996 | 1 | `F` |
| Report_Type | object | Field specific to Vis file | 996 | 3 | `VIS:VISIT`, `VIS:ED Update`, `VIS:ED Prov Note` |
| Report_Text | float64 | Full text of report | 1000 | 0 |  |

---

## Common Field Definitions

### Patient Identifiers
- **EMPI**: Enterprise Master Patient Identifier - Unique identifier for each patient across the Partners system
- **EPIC_PMRN**: EPIC system Patient Medical Record Number
- **MGH_MRN**: Massachusetts General Hospital Medical Record Number
- **BWH_MRN**: Brigham and Women's Hospital Medical Record Number
- **LMR**: Longitudinal Medical Record number

### Date/Time Fields
- Most date fields are in format: MM/DD/YYYY or MM/DD/YYYY HH:MM:SS
- Times are typically in 24-hour format

### Code Types
- **ICD-9**: International Classification of Diseases, 9th Revision
- **ICD-10**: International Classification of Diseases, 10th Revision
- **CPT**: Current Procedural Terminology codes
- **LOINC**: Logical Observation Identifiers Names and Codes (for lab tests)
- **RxNorm**: Standardized nomenclature for clinical drugs

### Common Flags and Indicators
- **Abnormal_Flag**: H (High), L (Low), N (Normal), A (Abnormal)
- **Report_Status**: F (Final), P (Preliminary), C (Corrected)
- **Inpatient_Outpatient**: I (Inpatient), O (Outpatient), E (Emergency)

---

## Data Quality Notes

1. **Patient Privacy**: All patient identifiers should be handled according to HIPAA guidelines
2. **Date Consistency**: Dates may vary in format across different files
3. **Missing Data**: Null values are common and should be handled appropriately
4. **Text Fields**: Free-text fields may contain PHI and should be processed carefully
5. **Code Versions**: Diagnosis and procedure codes may use different versions (ICD-9 vs ICD-10)

---

## File Relationships

The RPDR files are linked through common identifiers:

```
EMPI (Patient ID) -> Links all files for a specific patient
Encounter_Number -> Links visit-specific data (Enc, Dia, Prc, Med, Lab, etc.)
Report_ID -> Links reports with their details (Rad with Rdt, Lab with Lno)
```

## Usage Guidelines

1. **Loading Files**: Use pipe delimiter (|) when reading files
2. **Encoding**: Files typically use 'latin-1' encoding
3. **Large Files**: Some files may be very large; consider chunking or sampling
4. **Date Parsing**: Use appropriate date parsing for date columns
5. **Missing Values**: Handle various representations of null (empty string, 'NULL', etc.)

---

## Contact Information

For questions about RPDR data:
- Partners HealthCare RPDR Team
- Research Computing
- Data Governance Office

---

*This data dictionary was automatically generated from RPDR export files.*
