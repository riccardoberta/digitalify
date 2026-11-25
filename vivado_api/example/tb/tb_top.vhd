-- tb_top.vhd
-- Self-checking testbench for the top-level counter + LED driver design.
-- Verifies that after reset the LEDs track the counter value.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_top is
end entity tb_top;

architecture sim of tb_top is

    signal clk  : std_logic := '0';
    signal rst  : std_logic := '1';
    signal leds : std_logic_vector(3 downto 0);

    component top is
        port (
            clk  : in  std_logic;
            rst  : in  std_logic;
            leds : out std_logic_vector(3 downto 0)
        );
    end component top;

begin

    -- 100 MHz clock
    clk <= not clk after 5 ns;

    dut : top
        port map (
            clk  => clk,
            rst  => rst,
            leds => leds
        );

    stim : process
        variable expected : std_logic_vector(3 downto 0);
    begin
        -- Hold reset for 4 clock cycles
        wait for 40 ns;
        rst <= '0';

        -- Let the counter run for 20 cycles and check each output value
        for i in 0 to 19 loop
            wait until rising_edge(clk);
            wait for 1 ns;  -- small delta to let outputs settle
            -- After N cycles with rst=0, counter = N+1 (first increment at cycle 0)
            expected := std_logic_vector(to_unsigned((i + 1) mod 16, expected'length));
            assert leds = expected
                report "Unexpected LED value at cycle " & integer'image(i)
                severity error;
        end loop;

        report "Simulation completed successfully." severity note;
        wait;
    end process stim;

end architecture sim;
