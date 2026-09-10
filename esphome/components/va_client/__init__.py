import esphome.codegen as cg
import esphome.config_validation as cv
from esphome import automation
from esphome.components import microphone, speaker, micro_wake_word, binary_sensor, media_player
from esphome.components import esp32
from esphome.const import CONF_ID, CONF_URL, CONF_TRIGGER_ID
from esphome.core import CORE
import esphome.final_validate as fv

CODEOWNERS = ["@maxmaxme"]
DEPENDENCIES = ["network", "microphone", "speaker"]

CONF_MICROPHONE = "microphone"
CONF_MIC_CHANNEL = "mic_channel"
CONF_SPEAKER = "speaker"
CONF_BARGE_IN = "barge_in"
CONF_ON_PHASE = "on_phase"
CONF_ON_REPEATED_FAILURE = "on_repeated_failure"
CONF_ON_FOLLOWUP_OPENED = "on_followup_opened"
CONF_WAKE_RECORDING = "wake_recording"
CONF_WAKE_WORD = "wake_word"
CONF_MUTE_SENSOR = "mute_sensor"
CONF_MEDIA_PLAYER = "media_player"
CONF_SOURCE_ID = "source_id"

va_client_ns = cg.esphome_ns.namespace("va_client")
VaClient = va_client_ns.class_("VaClient", cg.Component)
OnPhaseTrigger = va_client_ns.class_(
    "OnPhaseTrigger", automation.Trigger.template(cg.std_string)
)
OnRepeatedFailureTrigger = va_client_ns.class_(
    "OnRepeatedFailureTrigger", automation.Trigger.template()
)
OnFollowupOpenedTrigger = va_client_ns.class_(
    "OnFollowupOpenedTrigger", automation.Trigger.template()
)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(VaClient),
        cv.Required(CONF_URL): cv.string,
        cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
        cv.Optional(CONF_MIC_CHANNEL, default=0): cv.int_range(min=0, max=1),
        cv.Optional(CONF_BARGE_IN, default=True): cv.boolean,
        cv.Required(CONF_SPEAKER): cv.use_id(speaker.Speaker),
        cv.Optional(CONF_WAKE_RECORDING): cv.Schema({
            cv.GenerateID(CONF_SOURCE_ID): cv.declare_id(microphone.MicrophoneSource),
            cv.Required(CONF_WAKE_WORD): cv.use_id(micro_wake_word.MicroWakeWord),
            cv.Required(CONF_MUTE_SENSOR): cv.use_id(binary_sensor.BinarySensor),
            cv.Required(CONF_MEDIA_PLAYER): cv.use_id(media_player.MediaPlayer),
        }),
        cv.Optional(CONF_ON_PHASE): automation.validate_automation(
            {
                cv.GenerateID(CONF_TRIGGER_ID): cv.declare_id(OnPhaseTrigger),
            }
        ),
        cv.Optional(CONF_ON_REPEATED_FAILURE): automation.validate_automation(
            {
                cv.GenerateID(CONF_TRIGGER_ID): cv.declare_id(OnRepeatedFailureTrigger),
            }
        ),
        cv.Optional(CONF_ON_FOLLOWUP_OPENED): automation.validate_automation(
            {
                cv.GenerateID(CONF_TRIGGER_ID): cv.declare_id(OnFollowupOpenedTrigger),
            }
        ),
    }
).extend(cv.COMPONENT_SCHEMA)


def recording_wake_source(config, full):
    wake = full.get("micro_wake_word")
    recording = config[CONF_WAKE_RECORDING]
    if not isinstance(wake, dict) or str(wake[CONF_ID]) != str(recording[CONF_WAKE_WORD]):
        raise cv.Invalid("wake_recording requires the configured micro_wake_word instance")
    source = wake[CONF_MICROPHONE]
    if str(source[CONF_MICROPHONE]) != str(config[CONF_MICROPHONE]):
        raise cv.Invalid("wake_recording and va_client must share the same physical microphone")
    if source["bits_per_sample"] != 16 or len(source["channels"]) != 1:
        raise cv.Invalid("wake_recording requires the mono PCM16 wake source")
    return source


def final_validate(config):
    if CONF_WAKE_RECORDING not in config:
        return config
    full = fv.full_config.get()
    if not full.get("api", {}).get("encryption", {}).get("key"):
        raise cv.Invalid("wake_recording controls require native API encryption")
    source = recording_wake_source(config, full)
    microphone.final_validate_microphone_source_schema("wake_recording", sample_rate=16000)(source)
    return config


FINAL_VALIDATE_SCHEMA = final_validate


async def to_code(config):
    # esp-idf managed component providing esp_websocket_client.
    esp32.add_idf_component(
        name="espressif/esp_websocket_client",
        ref="1.7.0",
    )

    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_url(config[CONF_URL]))
    cg.add(var.set_mic_channel(config[CONF_MIC_CHANNEL]))
    cg.add(var.set_barge_in(config[CONF_BARGE_IN]))

    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    spk = await cg.get_variable(config[CONF_SPEAKER])
    cg.add(var.set_speaker(spk))

    if recording := config.get(CONF_WAKE_RECORDING):
        # Derive from the FINAL mWW source, including package overrides. The
        # CRNN measurement package uses channel0/gain1; realtime uses1/gain4.
        # There is no independently configurable recording DSP path to drift.
        source = recording_wake_source(config, CORE.config)
        source_var = cg.new_Pvariable(
            recording[CONF_SOURCE_ID], mic, 16, source["gain_factor"], False
        )
        cg.add(source_var.add_channel(source["channels"][0]))
        cg.add_define("USE_PIPPA_WEB_CAPTURE")
        cg.add(var.set_recording_source(source_var))
        cg.add(var.set_recording_format(source["channels"][0], source["gain_factor"]))
        cg.add(var.set_recording_microphone_id(str(source[CONF_MICROPHONE])))
        cg.add(var.set_recording_wake_word(await cg.get_variable(recording[CONF_WAKE_WORD])))
        cg.add(var.set_recording_wake_source(await cg.get_variable(source[CONF_ID])))
        cg.add(var.set_recording_mute_sensor(await cg.get_variable(recording[CONF_MUTE_SENSOR])))
        cg.add(var.set_recording_player(await cg.get_variable(recording[CONF_MEDIA_PLAYER])))

    for conf in config.get(CONF_ON_PHASE, []):
        trigger = cg.new_Pvariable(conf[CONF_TRIGGER_ID], var)
        await automation.build_automation(trigger, [(cg.std_string, "phase")], conf)

    for conf in config.get(CONF_ON_REPEATED_FAILURE, []):
        trigger = cg.new_Pvariable(conf[CONF_TRIGGER_ID], var)
        await automation.build_automation(trigger, [], conf)

    for conf in config.get(CONF_ON_FOLLOWUP_OPENED, []):
        trigger = cg.new_Pvariable(conf[CONF_TRIGGER_ID], var)
        await automation.build_automation(trigger, [], conf)
