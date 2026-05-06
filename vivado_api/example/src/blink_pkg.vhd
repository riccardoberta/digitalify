
-- ============================================================================
--
-- This package contains shared constants and type definitions used across
-- multiple VHDL modules in the design.
--
-- The purpose of using a package is to centralize configuration parameters
-- and improve maintainability of the project. By defining common parameters
-- in a single location, all dependent modules can remain consistent without
-- duplicating definitions.
--
-- COUNTER_WIDTH defines the default bit width used for internal counters
-- within the design. Increasing this value produces slower visible blinking
-- rates because larger counters require more clock cycles to overflow.
--
-- Packages are commonly used in larger FPGA and ASIC projects to share:
--
--   - constants
--   - type definitions
--   - utility functions
--   - configuration parameters
--
-- across multiple source files.
-- ============================================================================

library ieee;
use ieee.std_logic_1164.all;

package blink_pkg is
    constant COUNTER_WIDTH : integer := 24;
end package;
