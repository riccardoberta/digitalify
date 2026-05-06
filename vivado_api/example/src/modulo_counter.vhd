-- ============================================================================
--
-- This module implements a synchronous up-counter with programmable modulo
-- value and clock-enable control.
--
-- The counter increments on the rising edge of the clock only when the
-- 'enable' signal is asserted. When the counter reaches MODULO-1, it wraps
-- back to zero, implementing cyclic counting behavior.
--
-- Main features:
--
--   - synchronous reset
--   - enable-controlled counting
--   - programmable modulo value through a generic
--   - unsigned arithmetic using ieee.numeric_std
--
-- Inputs:
--
--   clk
--       System clock. Counter updates occur on rising edges.
--
--   rst
--       Active-high synchronous reset. Forces the counter to zero.
--
--   enable
--       Clock-enable signal. Counting occurs only when asserted.
--
-- Outputs:
--
--   value
--       Current counter value represented as an unsigned vector.
--
-- Design notes:
--
-- The counter internally uses the unsigned type from numeric_std to enable
-- synthesizable arithmetic operations such as incrementing with '+'.
--
-- The modulo behavior is implemented by explicitly checking when the counter
-- reaches MODULO-1 before wrapping to zero.
--
-- This type of counter is commonly used for:
--
--   - frequency division
--   - event counting
--   - timing generation
--   - finite-state timing control
--   - LED blinking applications
--
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity modulo_counter is
    generic(
        MODULO : integer := 16
    );
    port(
        clk   : in  std_logic;
        rst   : in  std_logic;
        enable: in  std_logic;
        value : out unsigned(3 downto 0)
    );
end entity;

architecture rtl of modulo_counter is
    signal cnt : unsigned(3 downto 0) := (others => '0');
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                cnt <= (others => '0');
            elsif enable = '1' then
                if cnt = MODULO-1 then
                    cnt <= (others => '0');
                else
                    cnt <= cnt + 1;
                end if;
            end if;
        end if;
    end process;

    value <= cnt;
end architecture;
