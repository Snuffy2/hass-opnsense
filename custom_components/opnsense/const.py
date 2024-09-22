"""The OPNsense component."""

from typing import Final

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, UnitOfInformation, UnitOfTime

VERSION = "v0.3.1"
DEFAULT_USERNAME = ""
DOMAIN = "opnsense"
OPNSENSE_MIN_FIRMWARE = "24.7"

UNDO_UPDATE_LISTENER = "undo_update_listener"

PLATFORMS = ["sensor", "switch", "device_tracker", "binary_sensor", "update"]
LOADED_PLATFORMS = "loaded_platforms"

OPNSENSE_CLIENT = "opnsense_client"
COORDINATOR = "coordinator"
DEVICE_TRACKER_COORDINATOR = "device_tracker_coordinator"
SHOULD_RELOAD = "should_reload"
TRACKED_MACS = "tracked_macs"
DEFAULT_SCAN_INTERVAL = 30
CONF_TLS_INSECURE = "tls_insecure"
DEFAULT_TLS_INSECURE = False
DEFAULT_VERIFY_SSL = True

CONF_DEVICE_TRACKER_ENABLED = "device_tracker_enabled"
DEFAULT_DEVICE_TRACKER_ENABLED = False

CONF_DEVICE_TRACKER_SCAN_INTERVAL = "device_tracker_scan_interval"
DEFAULT_DEVICE_TRACKER_SCAN_INTERVAL = 150

CONF_DEVICE_TRACKER_CONSIDER_HOME = "device_tracker_consider_home"
DEFAULT_DEVICE_TRACKER_CONSIDER_HOME = 0

CONF_DEVICES = "devices"

COUNT = "count"

# pulled from upnp component
BYTES_RECEIVED = "bytes_received"
BYTES_SENT = "bytes_sent"
PACKETS_RECEIVED = "packets_received"
PACKETS_SENT = "packets_sent"
DATA_PACKETS = "packets"
DATA_RATE_PACKETS_PER_SECOND = f"{DATA_PACKETS}/{UnitOfTime.SECONDS}"

ICON_MEMORY = "mdi:memory"


SERVICE_CLOSE_NOTICE = "close_notice"
SERVICE_START_SERVICE = "start_service"
SERVICE_STOP_SERVICE = "stop_service"
SERVICE_RESTART_SERVICE = "restart_service"
SERVICE_SYSTEM_HALT = "system_halt"
SERVICE_SYSTEM_REBOOT = "system_reboot"
SERVICE_SEND_WOL = "send_wol"

DEFAULT_SERVICE_CLOSE_NOTICE_ID = "all"

ATTR_CONFIG = "config"
ATTR_FILESYSTEMS = "filesystems"
ATTR_INTERFACE = "interface"
ATTR_INTERFACES = "interfaces"
ATTR_MAC = "mac"
ATTR_MEMORY = "memory"
ATTR_ONLY_IF_RUNNING = "only_if_running"
ATTR_PREVIOUS_STATE = "previous_state"
ATTR_SERVICE_ID = "service_id"
ATTR_SERVICE_NAME = "service_name"
ATTR_SERVICE_TYPE = "service_type"
ATTR_SERVICES = "services"
ATTR_SYSTEM_INFO = "system_info"
ATTR_STATUS = "status"
ATTR_SYSTEM = "system"
ATTR_TELEMETRY = "telemetry"

SENSOR_TYPES: Final[dict[str, SensorEntityDescription]] = {
    # pfstate
    f"{ATTR_TELEMETRY}.pfstate.used": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.pfstate.used",
        name="pf State Table Used",
        native_unit_of_measurement=COUNT,
        icon="mdi:table-network",
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.pfstate.total": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.pfstate.total",
        name="pf State Table Total",
        native_unit_of_measurement=COUNT,
        icon="mdi:table-network",
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.pfstate.used_percent": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.pfstate.used_percent",
        name="pf State Table Used Percentage",
        native_unit_of_measurement=PERCENTAGE,
        icon="mdi:table-network",
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # mbuf
    f"{ATTR_TELEMETRY}.mbuf.used": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.mbuf.used",
        name="Memory Buffers Used",
        native_unit_of_measurement=UnitOfInformation.BYTES,
        icon=ICON_MEMORY,
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.mbuf.total": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.mbuf.total",
        name="Memory Buffers Total",
        native_unit_of_measurement=UnitOfInformation.BYTES,
        icon=ICON_MEMORY,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.mbuf.used_percent": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.mbuf.used_percent",
        name="Memory Buffers Used Percentage",
        native_unit_of_measurement=PERCENTAGE,
        icon=ICON_MEMORY,
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # memory with state_class due to being less static
    f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.swap_reserved": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.swap_reserved",
        name="Memory Swap Reserved",
        native_unit_of_measurement=UnitOfInformation.BYTES,
        icon=ICON_MEMORY,
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # memory without state_class due to being generally static
    f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.physmem": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.physmem",
        name="Memory Physmem",
        native_unit_of_measurement=UnitOfInformation.BYTES,
        icon=ICON_MEMORY,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.used": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.used",
        name="Memory Used",
        native_unit_of_measurement=UnitOfInformation.BYTES,
        icon=ICON_MEMORY,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.swap_total": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.swap_total",
        name="Memory Swap Total",
        native_unit_of_measurement=UnitOfInformation.BYTES,
        icon=ICON_MEMORY,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # memory percentages
    f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.swap_used_percent": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.swap_used_percent",
        name="Memory Swap Used Percentage",
        native_unit_of_measurement=PERCENTAGE,
        icon=ICON_MEMORY,
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.used_percent": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_MEMORY}.used_percent",
        name="Memory Used Percentage",
        native_unit_of_measurement=PERCENTAGE,
        icon=ICON_MEMORY,
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # cpu
    # "telemetry.cpu.frequency.current": SensorEntityDescription(
    #     key="telemetry.cpu.frequency.current",
    #     name="CPU Frequency Current",
    #     native_unit_of_measurement=UnitOfFrequency.HERTZ,
    #     icon="mdi:speedometer-medium",
    #     state_class=SensorStateClass.MEASUREMENT,
    #     # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    # ),
    # "telemetry.cpu.frequency.max": SensorEntityDescription(
    #     key="telemetry.cpu.frequency.max",
    #     name="CPU Frequency Max",
    #     native_unit_of_measurement=UnitOfFrequency.HERTZ,
    #     icon="mdi:speedometer",
    #     # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    # ),
    f"{ATTR_TELEMETRY}.cpu.count": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.cpu.count",
        name="CPU Count",
        native_unit_of_measurement=COUNT,
        icon="mdi:speedometer-medium",
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.cpu.usage_total": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.cpu.usage_total",
        name="CPU Usage",
        native_unit_of_measurement=PERCENTAGE,
        icon="mdi:speedometer-medium",
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.load_average.one_minute": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.load_average.one_minute",
        name="System Load Average One Minute",
        # native_unit_of_measurement=PERCENTAGE,
        icon="mdi:speedometer-slow",
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.load_average.five_minute": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.load_average.five_minute",
        name="System Load Average Five Minute",
        # native_unit_of_measurement=PERCENTAGE,
        icon="mdi:speedometer-slow",
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.load_average.fifteen_minute": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.load_average.fifteen_minute",
        name="System Load Average Fifteen Minute",
        # native_unit_of_measurement=PERCENTAGE,
        icon="mdi:speedometer-slow",
        state_class=SensorStateClass.MEASUREMENT,
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # system
    # "telemetry.system.temp": SensorEntityDescription(
    #    key="telemetry.system.temp",
    #    name="System Temperature",
    #    native_unit_of_measurement=UnitOfTemperature.CELSIUS,
    #    device_class=SensorDeviceClass.TEMPERATURE,
    #    icon="mdi:thermometer",
    #    state_class=SensorStateClass.MEASUREMENT,
    #    # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    # ),
    f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.boottime": SensorEntityDescription(
        key=f"{ATTR_TELEMETRY}.{ATTR_SYSTEM}.boottime",
        name="System Boottime",
        # native_unit_of_measurement=UnitOfTime.SECONDS,
        device_class=SensorDeviceClass.TIMESTAMP,
        icon="mdi:clock-outline",
        # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    ),
    # dhcp
    # "dhcp_stats.leases.total": SensorEntityDescription(
    #    key="dhcp_stats.leases.total",
    #    name="DHCP Leases Total",
    #    native_unit_of_measurement="clients",
    #    icon="mdi:ip-network-outline",
    #    state_class=SensorStateClass.MEASUREMENT,
    #    # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    # ),
    # "dhcp_stats.leases.online": SensorEntityDescription(
    #    key="dhcp_stats.leases.online",
    #    name="DHCP Leases Online",
    #    native_unit_of_measurement="clients",
    #    icon="mdi:ip-network-outline",
    #    state_class=SensorStateClass.MEASUREMENT,
    #    # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    # ),
    # "dhcp_stats.leases.offline": SensorEntityDescription(
    #    key="dhcp_stats.leases.offline",
    #    name="DHCP Leases Offline",
    #    native_unit_of_measurement="clients",
    #    icon="mdi:ip-network-outline",
    #    state_class=SensorStateClass.MEASUREMENT,
    #    # entity_category=ENTITY_CATEGORY_DIAGNOSTIC,
    # ),
}
