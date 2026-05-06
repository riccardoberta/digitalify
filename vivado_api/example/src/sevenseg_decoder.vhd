-- ============================================================================
-- Seven-segment display decoder.
--
-- This module converts a 4-bit unsigned binary value into the corresponding
-- seven-segment display control pattern.
--
-- The decoder is implemented as a purely combinational circuit using a
-- case statement. Whenever the input value changes, the output segment pattern
-- is immediately updated.
--
-- Inputs:
--
--   value
--       Four-bit unsigned input representing the digit to display.
--
-- Outputs:
--
--   seg
--       Seven-segment control vector.
--
-- Segment encoding:
--
-- The output vector controls the seven segments of a display:
--
--       ---0---
--      |       |
--      5       1
--      |       |
--       ---6---
--      |       |
--      4       2
--      |       |
--       ---3---
--
-- The implementation assumes an active-low seven-segment display:
--
--   '0' -> segment ON
--   '1' -> segment OFF
--
-- Currently implemented digits:
--
--   0 -> "1000000"
--   1 -> "1111001"
--   2 -> "0100100"
--   3 -> "0110000"
--
-- Any unsupported input value activates the default pattern:
--
--   "1111111"
--
-- which turns all segments off.
--
-- This module is commonly used in FPGA demonstration projects to visualize
-- internal counter values and state-machine outputs.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sevenseg_decoder is
    port(
        value : in  unsigned(3 downto 0);
        seg   : out std_logic_vector(6 downto 0)
    );
end entity;

architecture rtl of sevenseg_decoder is
begin
    process(value)
    begin
        case value is
            when "0000" => seg <= "1000000";
            when "0001" => seg <= "1111001";
            when "0010" => seg <= "0100100";
            when "0011" => seg <= "0110000";
            when others => seg <= "1111111";
        end case;
    end process;
end architecture;
