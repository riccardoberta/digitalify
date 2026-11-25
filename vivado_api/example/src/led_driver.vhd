-- led_driver.vhd
-- Drives four active-high LEDs from a 4-bit data bus.

library ieee;
use ieee.std_logic_1164.all;

entity led_driver is
    port (
        data : in  std_logic_vector(3 downto 0);
        leds : out std_logic_vector(3 downto 0)
    );
end entity led_driver;

architecture rtl of led_driver is
begin
    leds <= data;   -- Basys3 LEDs are active-high.
end architecture rtl;
