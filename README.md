# UMC UTRECHT HOSPITAL: SEPSIS EARLY WARNING FOR PRETERM INFANTS
## INTRODUCTION
The data science team in UMC [Utrecht hospital](https://www.umcutrecht.nl/en) led by [Annemarie van 't Veen](https://www.linkedin.com/posts/annemarie-van-t-veen-5b62719_ai4health-studentchallenge-digitalhealth-activity-6947080636947943424-h13F?utm_source=linkedin_share&utm_medium=member_desktop_web), collaborated with [FruitPunch AI](https://www.fruitpunch.ai/), to develop an early warning model that would predict the onset of sepsis "acute infection" in preterm newborns, hospitalized in the neonatal intensive care unit (NICU).
This repo showcases the solution developed within the span of this project.

## PROBLEM STATMENT
The medical subject matter experts (SMEs) at UMC Utrecht children hospital [(WKZ)](https://www.hetwkz.nl/en), explain that preterm infants are more vulnurable to acute infections (sepsis) during their incubation term, which would lead to a higher infant mortality rate. The neonatal intensive care staff would benefit from an early warning system that alarms them to a possible sepsis case before it's onset, with an elaborate time frame for intervention. To this end, UMC Utrecht's data science team is interested in developing a model that predicts the onset of sepsis in preterm infants during incubation using the infant's record of physiological datastreams.

In this project, I've developed an XGboost timeseries classification model that predicts the onset of sepsis in preterm infants, within a 12 hours prediction horizon.

## DATA
UMC Utrecht data team provided an electronic health record (EHR) which included profile data, such as: ***{gender, gestation age}***. UMC Utrecht also provided a timeseries database recording 13 physiological markers, with `1` minute resolution. The physio-markers included: ***{Heart rate, Respiratory rate, Systolic blood pressure, Diabolic blood pressure, Rectal temperature, Incubator temperature,...etc}***. The database recorded the timestamps at which notable events occured in an ``event`` feature. Notable events included: ***{Birth, Discharge, Positive blood culture, Negative blood culture, ...}***.

The training features included 10 out of 13 physilogical features, plus 2 profile features, namely; ``gender`` and ``gestation age``. The targets used in model training were derived from the `event` feature. The postive label was set to the timestamp that corresponded to a `positive blood culture` event. A postive blood culture timestamp recorded the time at which the medical staff became ***suspicious*** of sepsis development in an infant, thus ordering a blood culture, which happened to be positive.

Below is the subset of features that were used in model training.

