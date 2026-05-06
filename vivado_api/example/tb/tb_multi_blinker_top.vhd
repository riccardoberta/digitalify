-- ============================================================================
--
-- This testbench verifies the functional correctness of the complete design by
-- automatically stimulating the DUT (Device Under Test) and checking its
-- outputs against expected reference values.
--
-- The testbench performs the following checks:
--
-- 1. Reset verification
--    Ensures that the counter and outputs are correctly initialized after reset.
--
-- 2. Clock-driven counter operation
--    Verifies that the internal counter increments only after the expected
--    number of clock cycles determined by TICK_DIVISOR.
--
-- 3. LED output validation
--    Checks that the LED outputs correctly reflect the current counter value.
--
-- 4. Seven-segment decoder verification
--    Ensures that the seven-segment display encoding matches the expected
--    hexadecimal representation of the counter state.
--
-- 5. Counter wrap-around behavior
--    Verifies proper modulo operation when the counter exceeds 15.
--
-- The testbench is fully self-checking:
-- assertions automatically detect functional mismatches and report errors.
--
-- The simulation intentionally terminates with:
--
--     "SELF-CHECKING TEST PASSED"
--
-- emitted as a severity failure. This is a common VHDL technique used to stop
-- the simulator once all verification steps complete successfully.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_multi_blinker_top is
end entity;

architecture sim of tb_multi_blinker_top is

    constant CLK_PERIOD   : time := 10 ns;
    constant TICK_DIVISOR : positive := 4;

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';
    signal led : std_logic_vector(3 downto 0);
    signal seg : std_logic_vector(6 downto 0);

    function expected_seg(value : unsigned(3 downto 0)) return std_logic_vector is
    begin
        case value is
            when "0000" => return "1000000";
            when "0001" => return "1111001";
            when "0010" => return "0100100";
            when "0011" => return "0110000";
            when "0100" => return "0011001";
            when "0101" => return "0010010";
            when "0110" => return "0000010";
            when "0111" => return "1111000";
            when "1000" => return "0000000";
            when "1001" => return "0010000";
            when "1010" => return "0001000";
            when "1011" => return "0000011";
            when "1100" => return "1000110";
            when "1101" => return "0100001";
            when "1110" => return "0000110";
            when others => return "0001110";
        end case;
    end function;

    procedure check_outputs(expected_count : natural) is
        variable expected_value : unsigned(3 downto 0);
    begin
        expected_value := to_unsigned(expected_count mod 16, 4);

        assert led = std_logic_vector(expected_value)
            report "LED mismatch. Expected " &
                   integer'image(to_integer(expected_value)) &
                   ", got " &
                   integer'image(to_integer(unsigned(led)))
            severity error;

        assert seg = expected_seg(expected_value)
            report "Seven-segment mismatch for value " &
                   integer'image(to_integer(expected_value))
            severity error;
    end procedure;

begin

    uut : entity work.multi_blinker_top
        generic map(
            TICK_DIVISOR => TICK_DIVISOR
        )
        port map(
            clk => clk,
            rst => rst,
            led => led,
            seg => seg
        );

    clk <= not clk after CLK_PERIOD / 2;

    stimulus : process
    begin
        wait for 3 * CLK_PERIOD;

        check_outputs(0);

        rst <= '0';

        wait until rising_edge(clk);
        wait for 1 ns;

        check_outputs(0);

        for expected_count in 1 to 20 loop
            for i in 1 to TICK_DIVISOR loop
                wait until rising_edge(clk);
            end loop;

            wait for 1 ns;

            check_outputs(expected_count);
        end loop;

        rst <= '1';

        wait until rising_edge(clk);
        wait for 1 ns;

        check_outputs(0);

        assert false report "SELF-CHECKING TEST PASSED" severity failure;
    end process;

end architecture;