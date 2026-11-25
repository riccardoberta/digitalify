-- top.vhd
-- Top-level entity: wires a 4-bit counter to the LED driver.

library ieee;
use ieee.std_logic_1164.all;

entity top is
    port (
        clk  : in  std_logic;
        rst  : in  std_logic;
        leds : out std_logic_vector(3 downto 0)
    );
end entity top;

architecture rtl of top is

    component counter is
        port (
            clk   : in  std_logic;
            rst   : in  std_logic;
            count : out std_logic_vector(3 downto 0)
        );
    end component counter;

    component led_driver is
        port (
            data : in  std_logic_vector(3 downto 0);
            leds : out std_logic_vector(3 downto 0)
        );
    end component led_driver;

    signal count : std_logic_vector(3 downto 0);

begin

    cnt_inst : counter
        port map (
            clk   => clk,
            rst   => rst,
            count => count
        );

    led_inst : led_driver
        port map (
            data => count,
            leds => leds
        );

end architecture rtl;
