-- ============================================================================
--
-- This module generates a periodic single-clock-cycle pulse used as a timing
-- enable signal for slower sequential logic.
--
-- Internally, the module implements a counter that increments on every rising
-- edge of the input clock. When the counter reaches DIVISOR-1, it resets to
-- zero and produces a one-cycle pulse on the 'tick' output.
--
-- Inputs:
--
--   clk
--       System clock.
--
--   rst
--       Active-high synchronous reset.
--
-- Outputs:
--
--   tick
--       Single-clock-cycle pulse generated periodically according to DIVISOR.
--
-- Generic parameters:
--
--   DIVISOR
--       Number of clock cycles between consecutive tick pulses.
--
-- Functional behavior:
--
-- For a clock frequency Fclk, the output tick frequency is approximately:
--
--       Ftick = Fclk / DIVISOR
--
-- For example:
--
--   Fclk    = 100 MHz
--   DIVISOR = 5,000,000
--
-- produces:
--
--   Ftick = 20 Hz
--
-- meaning one tick pulse every 50 ms.
--
-- Design notes:
--
-- The generated pulse remains asserted for exactly one clock cycle.
-- This approach is commonly preferred over generating derived clocks because
-- it avoids clock-domain creation and simplifies FPGA timing closure.
--
-- Typical applications include:
--
--   - LED blinking
--   - clock-enable generation
--   - timing control
--   - low-frequency event scheduling
--   - finite-state machine pacing
--
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tick_generator is
    generic(
        DIVISOR : integer := 5000000
    );
    port(
        clk  : in  std_logic;
        rst  : in  std_logic;
        tick : out std_logic
    );
end entity;

architecture rtl of tick_generator is
    signal counter : integer range 0 to DIVISOR-1 := 0;
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                counter <= 0;
                tick <= '0';
            else
                if counter = DIVISOR-1 then
                    counter <= 0;
                    tick <= '1';
                else
                    counter <= counter + 1;
                    tick <= '0';
                end if;
            end if;
        end if;
    end process;
end architecture;
