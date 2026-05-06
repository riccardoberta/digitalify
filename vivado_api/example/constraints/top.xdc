
# ============================================================================
# FPGA pin assignment and I/O electrical constraints file (XDC).
#
# This file maps the top-level VHDL ports of the design to the physical pins
# of the FPGA package and specifies the electrical I/O standard used by those
# pins.
#
# In Vivado projects, XDC (Xilinx Design Constraints) files are essential
# because the HDL code itself only describes logical behavior and does not
# specify how signals connect to the actual FPGA board hardware.
#
# ----------------------------------------------------------------------------
# Clock input assignment
# ----------------------------------------------------------------------------
#
# set_property PACKAGE_PIN W5 [get_ports clk]
#
#   Maps the VHDL signal 'clk' to the physical FPGA package pin W5.
#
#   On a development board, this pin is typically connected to the onboard
#   oscillator that provides the system clock.
#
# set_property IOSTANDARD LVCMOS33 [get_ports clk]
#
#   Configures the electrical signaling standard for the clock pin.
#
#   LVCMOS33 means:
#
#       - Low Voltage CMOS
#       - 3.3 V signaling level
#
#   The selected I/O standard must match the electrical characteristics of
#   the external hardware connected to the FPGA pin.
#
# ----------------------------------------------------------------------------
# LED output assignments
# ----------------------------------------------------------------------------
#
# set_property PACKAGE_PIN U16 [get_ports {led[0]}]
# set_property PACKAGE_PIN E19 [get_ports {led[1]}]
# set_property PACKAGE_PIN U19 [get_ports {led[2]}]
# set_property PACKAGE_PIN V19 [get_ports {led[3]}]
#
#   These constraints map each LED output signal to a specific physical FPGA
#   pin connected to onboard LEDs on the FPGA development board.
#
#   The mapping is:
#
#       led[0] -> U16
#       led[1] -> E19
#       led[2] -> U19
#       led[3] -> V19
#
# ----------------------------------------------------------------------------
# LED electrical standard
# ----------------------------------------------------------------------------
#
# set_property IOSTANDARD LVCMOS33 [get_ports led[*]]
#
#   Applies the LVCMOS33 electrical standard to all signals belonging to the
#   LED bus:
#
#       led[0]
#       led[1]
#       led[2]
#       led[3]
#
#   The wildcard syntax:
#
#       led[*]
#
#   allows a single constraint to be applied to every bit of the vector.
#
# ----------------------------------------------------------------------------
# Why constraints are necessary
# ----------------------------------------------------------------------------
#
# Without an XDC file:
#
#   - Vivado would not know which FPGA pins to use
#   - synthesis could complete, but implementation would fail
#   - the generated bitstream would not correctly interface with the board
#
# Constraint files therefore provide the physical hardware integration layer
# between the logical HDL design and the FPGA board.
#
# ============================================================================

set_property PACKAGE_PIN W5 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]

set_property PACKAGE_PIN U16 [get_ports {led[0]}]
set_property PACKAGE_PIN E19 [get_ports {led[1]}]
set_property PACKAGE_PIN U19 [get_ports {led[2]}]
set_property PACKAGE_PIN V19 [get_ports {led[3]}]

set_property IOSTANDARD LVCMOS33 [get_ports led[*]]
