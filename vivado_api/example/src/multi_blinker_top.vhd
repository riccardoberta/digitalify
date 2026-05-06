-- ============================================================================
--
-- This entity connects three lower-level components into a complete FPGA
-- demonstration circuit:
--
--   1. tick_generator
--      Produces a periodic one-clock-cycle enable pulse.
--
--   2. modulo_counter
--      Increments a 4-bit counter each time the tick pulse is asserted.
--
--   3. sevenseg_decoder
--      Converts the current counter value into a seven-segment display pattern.
--
-- The current counter value is also directly mapped to the four LED outputs,
-- so the LEDs show the binary representation of the counter.
--
-- Inputs:
--
--   clk
--       System clock.
--
--   rst
--       Active-high synchronous reset distributed to the internal modules.
--
-- Outputs:
--
--   led
--       Four-bit LED output showing the current counter value in binary.
--
--   seg
--       Seven-segment display output showing the same counter value as a
--       hexadecimal digit.
--
-- Design behavior:
--
-- The tick generator divides the input clock by the configured DIVISOR value.
-- Each generated tick enables the modulo counter for exactly one clock cycle.
-- The counter cycles from 0 to 15 and then wraps back to 0. Its value drives
-- both the LEDs and the seven-segment decoder.
--
-- This file acts as the structural integration point of the design and is the
-- top entity that should be selected for synthesis.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multi_blinker_top is
    port(
        clk : in  std_logic;
        rst : in  std_logic;
        led : out std_logic_vector(3 downto 0);
        seg : out std_logic_vector(6 downto 0)
    );
end entity;

architecture rtl of multi_blinker_top is

    signal tick_sig : std_logic;
    signal count    : unsigned(3 downto 0);

begin

    tick_inst : entity work.tick_generator
        generic map(
            DIVISOR => 10
        )
        port map(
            clk  => clk,
            rst  => rst,
            tick => tick_sig
        );

    counter_inst : entity work.modulo_counter
        generic map(
            MODULO => 16
        )
        port map(
            clk    => clk,
            rst    => rst,
            enable => tick_sig,
            value  => count
        );

    decoder_inst : entity work.sevenseg_decoder
        port map(
            value => count,
            seg   => seg
        );

    led <= std_logic_vector(count);

end architecture;
