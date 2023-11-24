import pathlib

# this is the dir the streamlit app is launched from
APP_DIR = pathlib.Path(__file__).parent.absolute()
LOCAL_DIR = APP_DIR / "local"
LOCAL_DIR_OUT = APP_DIR / "out"

# the name of a file for logo image
# to be put in APP_DIR
LOGO = "./oracle.png"

# the list of file type supported (and that can be loaded)
FILE_TYPE_SUPPORTED = ["txt"]

MAX_CHAR_DISPLAYED = 100

# parameters for Text Classification custom model
OCI_MODEL_ENDPOINT = "https://language.aiservice.eu-frankfurt-1.oci.oraclecloud.com"

OCI_MODEL_ID = "ocid1.ailanguagemodel.oc1.eu-frankfurt-1.amaaaaaangencdyawntujzg2ok2yknb6heuafgv7mjyxy3zwufp5lyl4zafa"

# per i modelli custom questo parametro è fondamentale
# model4
# OCI_ENDPOINT_ID = "ocid1.ailanguageendpoint.oc1.eu-frankfurt-1.amaaaaaangencdyaaj5ggyngtx5yyvm5zabsoqsgrojvnuz67edoylbzjtkq"

# This is the endpoint for model 5, last developed on 16/11
OCI_ENDPOINT_ID = "ocid1.ailanguageendpoint.oc1.eu-frankfurt-1.amaaaaaangencdyalugphcz233tait4usukfpivke64bbap767zgvhyrksha"

# Sentiment Analysis Model
HF_SENT_MODEL_NAME = "luigisaetta/sentiment_ita"