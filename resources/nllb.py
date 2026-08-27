#
# nllb.py
import gradio as gr
import os
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from .common import *
from modules.util import free_cuda_mem, free_cuda_cache
import traceback
model_path_nllb = "./models/nllb/"
os.makedirs(model_path_nllb, exist_ok=True)
translated_tokens = None

model_list_nllb = []

for file_path, _, file_list in os.walk(model_path_nllb):
    for filename in file_list:
        f = os.path.join(file_path, filename)
        if os.path.isfile(f) and (filename.endswith('.ckpt') or filename.endswith('.bin') or filename.endswith('.safetensors')):
            model_list_nllb.append(os.path.dirname(f))

model_list_nllb_builtin = [
    "facebook/nllb-200-distilled-600M",
]

for k in range(len(model_list_nllb_builtin)):
    model_list_nllb.append(model_list_nllb_builtin[k])

# Liste des langues
language_list_nllb = {
    "Acehnese (Arabic script) 	 	    ": "ace_Arab",
    "Acehnese (Latin script) 	 	 	": "ace_Latn",
    "Mesopotamian Arabic 	 	 	 	": "acm_Arab",
    "Ta’izzi-Adeni Arabic 	 	 	 	": "acq_Arab",
    "Tunisian Arabic 	 	 	 	 	": "aeb_Arab",
    "Afrikaans 	 	 	 	 	 	 	": "afr_Latn",
    "South Levantine Arabic 	 	 	": "ajp_Arab",
    "Akan 	 	 	 	 	 	 	 	": "aka_Latn",
    "Amharic 	 	 	 	 	 	 	": "amh_Ethi",
    "North Levantine Arabic 	 	 	": "apc_Arab",
    "Modern Standard Arabic 	 	 	": "arb_Arab",
    "Modern Standard Arabic (Romanized) ": "arb_Latn",
    "Najdi Arabic 	 	 	 	 	 	": "ars_Arab",
    "Moroccan Arabic 	 	 	 	 	": "ary_Arab",
    "Egyptian Arabic 	 	 	 	 	": "arz_Arab",
    "Assamese 	 	 	 	 	 	 	": "asm_Beng",
    "Asturian 	 	 	 	 	 	 	": "ast_Latn",
    "Awadhi 	 	 	 	 	 	 	": "awa_Deva",
    "Central Aymara 	 	 	 	 	": "ayr_Latn",
    "South Azerbaijani 	 	 	 	 	": "azb_Arab",
    "North Azerbaijani 	 	 	 	 	": "azj_Latn",
    "Bashkir 	 	 	 	 	 	 	": "bak_Cyrl",
    "Bambara 	 	 	 	 	 	 	": "bam_Latn",
    "Balinese 	 	 	 	 	 	 	": "ban_Latn",
    "Belarusian 	 	 	 	 	 	": "bel_Cyrl",
    "Bemba 	 	 	 	 	 	 	 	": "bem_Latn",
    "Bengali 	 	 	 	 	 	 	": "ben_Beng",
    "Bhojpuri 	 	 	 	 	 	 	": "bho_Deva",
    "Banjar (Arabic script) 	 	 	": "bjn_Arab",
    "Banjar (Latin script) 	 	 	 	": "bjn_Latn",
    "Standard Tibetan 	 	 	 	 	": "bod_Tibt",
    "Bosnian 	 	 	 	 	 	 	": "bos_Latn",
    "Buginese 	 	 	 	 	 	 	": "bug_Latn",
    "Bulgarian 	 	 	 	 	 	 	": "bul_Cyrl",
    "Catalan 	 	 	 	 	 	 	": "cat_Latn",
    "Cebuano 	 	 	 	 	 	 	": "ceb_Latn",
    "Czech 	 	 	 	 	 	 	 	": "ces_Latn",
    "Chokwe 	 	 	 	 	 	 	": "cjk_Latn",
    "Central Kurdish 	 	 	 	 	": "ckb_Arab",
    "Crimean Tatar 	 	 	 	 	 	": "crh_Latn",
    "Welsh 	 	 	 	 	 	 	 	": "cym_Latn",
    "Danish 	 	 	 	 	 	 	": "dan_Latn",
    "German 	 	 	 	 	 	 	": "deu_Latn",
    "Southwestern Dinka 	 	 	 	": "dik_Latn",
    "Dyula 	 	 	 	 	 	 	 	": "dyu_Latn",
    "Dzongkha 	 	 	 	 	 	 	": "dzo_Tibt",
    "Greek 	 	 	 	 	 	 	 	": "ell_Grek",
    "English 	 	 	 	 	 	 	": "eng_Latn",
    "Esperanto 	 	 	 	 	 	 	": "epo_Latn",
    "Estonian 	 	 	 	 	 	 	": "est_Latn",
    "Basque 	 	 	 	 	 	 	": "eus_Latn",
    "Ewe 	 	 	 	 	 	 	 	": "ewe_Latn",
    "Faroese 	 	 	 	 	 	 	": "fao_Latn",
    "Fijian 	 	 	 	 	 	 	": "fij_Latn",
    "Finnish 	 	 	 	 	 	 	": "fin_Latn",
    "Fon 	 	 	 	 	 	 	 	": "fon_Latn",
    "French 	 	 	 	 	 	 	": "fra_Latn",
    "Friulian 	 	 	 	 	 	 	": "fur_Latn",
    "Nigerian Fulfulde 	 	 	 	 	": "fuv_Latn",
    "Scottish Gaelic 	 	 	 	 	": "gla_Latn",
    "Irish 	 	 	 	 	 	 	 	": "gle_Latn",
    "Galician 	 	 	 	 	 	 	": "glg_Latn",
    "Guarani 	 	 	 	 	 	 	": "grn_Latn",
    "Gujarati 	 	 	 	 	 	 	": "guj_Gujr",
    "Haitian Creole 	 	 	 	 	": "hat_Latn",
    "Hausa 	 	 	 	 	 	 	 	": "hau_Latn",
    "Hebrew 	 	 	 	 	 	 	": "heb_Hebr",
    "Hindi 	 	 	 	 	 	 	 	": "hin_Deva",
    "Chhattisgarhi 	 	 	 	 	 	": "hne_Deva",
    "Croatian 	 	 	 	 	 	 	": "hrv_Latn",
    "Hungarian 	 	 	 	 	 	 	": "hun_Latn",
    "Armenian 	 	 	 	 	 	 	": "hye_Armn",
    "Igbo 	 	 	 	 	 	 	 	": "ibo_Latn",
    "Ilocano 	 	 	 	 	 	 	": "ilo_Latn",
    "Indonesian 	 	 	 	 	 	": "ind_Latn",
    "Icelandic 	 	 	 	 	 	 	": "isl_Latn",
    "Italian 	 	 	 	 	 	 	": "ita_Latn",
    "Javanese 	 	 	 	 	 	 	": "jav_Latn",
    "Japanese 	 	 	 	 	 	 	": "jpn_Jpan",
    "Kabyle 	 	 	 	 	 	 	": "kab_Latn",
    "Jingpho 	 	 	 	 	 	 	": "kac_Latn",
    "Kamba 	 	 	 	 	 	 	 	": "kam_Latn",
    "Kannada 	 	 	 	 	 	 	": "kan_Knda",
    "Kashmiri (Arabic script) 	 	 	": "kas_Arab",
    "Kashmiri (Devanagari script) 	 	": "kas_Deva",
    "Georgian 	 	 	 	 	 	 	": "kat_Geor",
    "Central Kanuri (Arabic script) 	": "knc_Arab",
    "Central Kanuri (Latin script) 	 	": "knc_Latn",
    "Kazakh 	 	 	 	 	 	 	": "kaz_Cyrl",
    "Kabiyè 	 	 	 	 	 	 	": "kbp_Latn",
    "Kabuverdianu 	 	 	 	 	 	": "kea_Latn",
    "Khmer 	 	 	 	 	 	 	 	": "khm_Khmr",
    "Kikuyu 	 	 	 	 	 	 	": "kik_Latn",
    "Kinyarwanda 	 	 	 	 	 	": "kin_Latn",
    "Kyrgyz 	 	 	 	 	 	 	": "kir_Cyrl",
    "Kimbundu 	 	 	 	 	 	 	": "kmb_Latn",
    "Northern Kurdish 	 	 	 	 	": "kmr_Latn",
    "Kikongo 	 	 	 	 	 	 	": "kon_Latn",
    "Korean 	 	 	 	 	 	 	": "kor_Hang",
    "Lao 	 	 	 	 	 	 	 	": "lao_Laoo",
    "Ligurian 	 	 	 	 	 	 	": "lij_Latn",
    "Limburgish 	 	 	 	 	 	": "lim_Latn",
    "Lingala 	 	 	 	 	 	 	": "lin_Latn",
    "Lithuanian 	 	 	 	 	 	": "lit_Latn",
    "Lombard 	 	 	 	 	 	 	": "lmo_Latn",
    "Latgalian 	 	 	 	 	 	 	": "ltg_Latn",
    "Luxembourgish 	 	 	 	 	 	": "ltz_Latn",
    "Luba-Kasai 	 	 	 	 	 	": "lua_Latn",
    "Ganda 	 	 	 	 	 	 	 	": "lug_Latn",
    "Luo 	 	 	 	 	 	 	 	": "luo_Latn",
    "Mizo 	 	 	 	 	 	 	 	": "lus_Latn",
    "Standard Latvian 	 	 	 	 	": "lvs_Latn",
    "Magahi 	 	 	 	 	 	 	": "mag_Deva",
    "Maithili 	 	 	 	 	 	 	": "mai_Deva",
    "Malayalam 	 	 	 	 	 	 	": "mal_Mlym",
    "Marathi 	 	 	 	 	 	 	": "mar_Deva",
    "Minangkabau (Arabic script) 	 	": "min_Arab",
    "Minangkabau (Latin script) 	 	": "min_Latn",
    "Macedonian 	 	 	 	 	 	": "mkd_Cyrl",
    "Plateau Malagasy 	 	 	 	 	": "plt_Latn",
    "Maltese 	 	 	 	 	 	 	": "mlt_Latn",
    "Meitei (Bengali script) 	 	 	": "mni_Beng",
    "Halh Mongolian 	 	 	 	 	": "khk_Cyrl",
    "Mossi 	 	 	 	 	 	 	 	": "mos_Latn",
    "Maori 	 	 	 	 	 	 	 	": "mri_Latn",
    "Burmese 	 	 	 	 	 	 	": "mya_Mymr",
    "Dutch 	 	 	 	 	 	 	 	": "nld_Latn",
    "Norwegian Nynorsk 	 	 	 	 	": "nno_Latn",
    "Norwegian Bokmål 	 	 	 	 	": "nob_Latn",
    "Nepali 	 	 	 	 	 	 	": "npi_Deva",
    "Northern Sotho 	 	 	 	 	": "nso_Latn",
    "Nuer 	 	 	 	 	 	 	 	": "nus_Latn",
    "Nyanja 	 	 	 	 	 	 	": "nya_Latn",
    "Occitan 	 	 	 	 	 	 	": "oci_Latn",
    "West Central Oromo 	 	 	 	": "gaz_Latn",
    "Odia 	 	 	 	 	 	 	 	": "ory_Orya",
    "Pangasinan 	 	 	 	 	 	": "pag_Latn",
    "Eastern Panjabi 	 	 	 	 	": "pan_Guru",
    "Papiamento 	 	 	 	 	 	": "pap_Latn",
    "Western Persian 	 	 	 	 	": "pes_Arab",
    "Polish 	 	 	 	 	 	 	": "pol_Latn",
    "Portuguese 	 	 	 	 	 	": "por_Latn",
    "Dari 	 	 	 	 	 	 	 	": "prs_Arab",
    "Southern Pashto 	 	 	 	 	": "pbt_Arab",
    "Ayacucho Quechua 	 	 	 	 	": "quy_Latn",
    "Romanian 	 	 	 	 	 	 	": "ron_Latn",
    "Rundi 	 	 	 	 	 	 	 	": "run_Latn",
    "Russian 	 	 	 	 	 	 	": "rus_Cyrl",
    "Sango 	 	 	 	 	 	 	 	": "sag_Latn",
    "Sanskrit 	 	 	 	 	 	 	": "san_Deva",
    "Santali 	 	 	 	 	 	 	": "sat_Olck",
    "Sicilian 	 	 	 	 	 	 	": "scn_Latn",
    "Shan 	 	 	 	 	 	 	 	": "shn_Mymr",
    "Sinhala 	 	 	 	 	 	 	": "sin_Sinh",
    "Slovak 	 	 	 	 	 	 	": "slk_Latn",
    "Slovenian 	 	 	 	 	 	 	": "slv_Latn",
    "Samoan 	 	 	 	 	 	 	": "smo_Latn",
    "Shona 	 	 	 	 	 	 	 	": "sna_Latn",
    "Sindhi 	 	 	 	 	 	 	": "snd_Arab",
    "Somali 	 	 	 	 	 	 	": "som_Latn",
    "Southern Sotho 	 	 	 	 	": "sot_Latn",
    "Spanish 	 	 	 	 	 	 	": "spa_Latn",
    "Tosk Albanian 	 	 	 	 	 	": "als_Latn",
    "Sardinian 	 	 	 	 	 	 	": "srd_Latn",
    "Serbian 	 	 	 	 	 	 	": "srp_Cyrl",
    "Swati 	 	 	 	 	 	 	 	": "ssw_Latn",
    "Sundanese 	 	 	 	 	 	 	": "sun_Latn",
    "Swedish 	 	 	 	 	 	 	": "swe_Latn",
    "Swahili 	 	 	 	 	 	 	": "swh_Latn",
    "Silesian 	 	 	 	 	 	 	": "szl_Latn",
    "Tamil 	 	 	 	 	 	 	 	": "tam_Taml",
    "Tatar 	 	 	 	 	 	 	 	": "tat_Cyrl",
    "Telugu 	 	 	 	 	 	 	": "tel_Telu",
    "Tajik 	 	 	 	 	 	 	 	": "tgk_Cyrl",
    "Tagalog 	 	 	 	 	 	 	": "tgl_Latn",
    "Thai 	 	 	 	 	 	 	 	": "tha_Thai",
    "Tigrinya 	 	 	 	 	 	 	": "tir_Ethi",
    "Tamasheq (Latin script) 	 	 	": "taq_Latn",
    "Tamasheq (Tifinagh script) 	 	": "taq_Tfng",
    "Tok Pisin 	 	 	 	 	 	 	": "tpi_Latn",
    "Tswana 	 	 	 	 	 	 	": "tsn_Latn",
    "Tsonga 	 	 	 	 	 	 	": "tso_Latn",
    "Turkmen 	 	 	 	 	 	 	": "tuk_Latn",
    "Tumbuka 	 	 	 	 	 	 	": "tum_Latn",
    "Turkish 	 	 	 	 	 	 	": "tur_Latn",
    "Twi 	 	 	 	 	 	 	 	": "twi_Latn",
    "Central Atlas Tamazight 	 	 	": "tzm_Tfng",
    "Uyghur 	 	 	 	 	 	 	": "uig_Arab",
    "Ukrainian 	 	 	 	 	 	 	": "ukr_Cyrl",
    "Umbundu 	 	 	 	 	 	 	": "umb_Latn",
    "Urdu 	 	 	 	 	 	 	 	": "urd_Arab",
    "Northern Uzbek 	 	 	 	 	": "uzn_Latn",
    "Venetian 	 	 	 	 	 	 	": "vec_Latn",
    "Vietnamese 	 	 	 	 	 	": "vie_Latn",
    "Waray 	 	 	 	 	 	 	 	": "war_Latn",
    "Wolof 	 	 	 	 	 	 	 	": "wol_Latn",
    "Xhosa 	 	 	 	 	 	 	 	": "xho_Latn",
    "Eastern Yiddish 	 	 	 	 	": "ydd_Hebr",
    "Yoruba 	 	 	 	 	 	 	": "yor_Latn",
    "Yue Chinese 	 	 	 	 	 	": "yue_Hant",
    "Chinese (Simplified) 	 	 	 	": "zho_Hans",
    "Chinese (Traditional) 	 	 	 	": "zho_Hant",
    "Standard Malay 	 	 	 	 	": "zsm_Latn",
    "Zulu 	 	 	 	 	 	 	 	": "zul_Latn",
}


@metrics_decoration
def text_nllb(
        modelid_nllb,
        max_tokens_nllb,
        source_language_nllb,
        prompt_nllb,
        output_language_nllb,
        progress_nllb=gr.Progress(track_tqdm=True)
):
    # 在函数开始时初始化所有变量
    output_nllb = None
    model_nllb = None
    tokenizer_nllb = None
    automodel_nllb = None
    inputs_nllb = None
    translated_tokens = None

    try:
        print(">>>[nllb translation 👥 ]: starting module")

        device_label_nllb, model_arch = detect_device()
        print(f"[device_label_nllb]: {device_label_nllb}  |   [arch]: {model_arch}")
        device_nllb = torch.device(device_label_nllb)

        source_language_nllb = language_list_nllb[source_language_nllb]
        output_language_nllb = language_list_nllb[output_language_nllb]

        if modelid_nllb in model_list_nllb_builtin:
            model_nllb = snapshot_download(
                repo_id=modelid_nllb,
                cache_dir=model_path_nllb,
                resume_download=True,
                local_files_only=True
            )
        else:
            model_nllb = modelid_nllb

        tokenizer_nllb = NllbTokenizer.from_pretrained(
            model_nllb,
            torch_dtype=model_arch,
            src_lang=source_language_nllb,
            tgt_lang=output_language_nllb
        )

        automodel_nllb = AutoModelForSeq2SeqLM.from_pretrained(model_nllb).to(device_nllb)
        inputs_nllb = tokenizer_nllb(prompt_nllb, return_tensors="pt").to(device_nllb)

        if hasattr(automodel_nllb, 'to_bettertransformer'):
            automodel_nllb = automodel_nllb.to_bettertransformer()
        else:
            print("Warning: to_bettertransformer not available, skipping optimization")

        # 修复：使用 convert_ids_to_tokens 或 get_vocab 获取语言 ID
        try:
            # 方法1：使用 tokenizer 的转换方法
            forced_bos_token_id = tokenizer_nllb.convert_tokens_to_ids(output_language_nllb)

            # 如果上面获取不到，尝试方法2
            if forced_bos_token_id is None:
                # 检查 tokenizer 是否有 lang_code_to_id 属性
                if hasattr(tokenizer_nllb, 'lang_code_to_id'):
                    forced_bos_token_id = tokenizer_nllb.lang_code_to_id[output_language_nllb]
                elif hasattr(tokenizer_nllb, 'added_tokens_decoder'):
                    # 从 added_tokens 中查找
                    for token_id, token in tokenizer_nllb.added_tokens_decoder.items():
                        if token == output_language_nllb:
                            forced_bos_token_id = token_id
                            break
                else:
                    # 最后尝试从 vocab 获取
                    forced_bos_token_id = tokenizer_nllb.get_vocab().get(output_language_nllb)

            if forced_bos_token_id is None:
                raise ValueError(f"Could not find token ID for language: {output_language_nllb}")

        except Exception as e:
            print(f"Warning: Could not get language token ID: {e}")
            # 如果无法获取语言 ID，不使用 forced_bos_token_id
            forced_bos_token_id = None

        # 根据是否有 forced_bos_token_id 调用 generate
        if forced_bos_token_id is not None:
            translated_tokens = automodel_nllb.generate(
                **inputs_nllb,
                forced_bos_token_id=forced_bos_token_id,
                max_new_tokens=max_tokens_nllb,
            )
        else:
            translated_tokens = automodel_nllb.generate(
                **inputs_nllb,
                max_new_tokens=max_tokens_nllb,
            )

        output_nllb = tokenizer_nllb.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        filename_nllb = write_file(output_nllb)

        print(f">>>[nllb translation 👥 ]: generated 1 translation")
        reporting_nllb = f">>>[nllb translation 👥 ]: Settings : Model={modelid_nllb} | Max tokens={max_tokens_nllb} | Source language={source_language_nllb} | Output language={output_language_nllb} | Prompt={prompt_nllb}"
        print(reporting_nllb)

    except Exception as e:
        print(f"Error in text_nllb: {e}")
        traceback.print_exc()

    finally:
        # 安全清理 - 只清理已定义的变量
        for var_name in ['model_nllb', 'tokenizer_nllb', 'automodel_nllb', 'inputs_nllb', 'translated_tokens']:
            var_value = locals().get(var_name)
            if var_value is not None:
                try:
                    del var_value
                except:
                    pass

        clean_ram()
        print(f">>>[nllb translation 👥 ]: leaving module")

    return output_nllb