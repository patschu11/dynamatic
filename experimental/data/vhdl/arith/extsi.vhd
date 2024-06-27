library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity extsi is
  generic (
    INPUT_WIDTH  : integer;
    OUTPUT_WIDTH : integer
  );
  port (
    clk, rst : in std_logic;
    -- input channel
    ins       : in  std_logic_vector(INPUT_WIDTH - 1 downto 0);
    ins_valid : in  std_logic;
    ins_ready : out std_logic;
    -- output channel
    outs       : out std_logic_vector(OUTPUT_WIDTH - 1 downto 0);
    outs_valid : out std_logic;
    outs_ready : in  std_logic
  );
end entity;

architecture arch of extsi is
begin
  outs(OUTPUT_WIDTH - 1 downto INPUT_WIDTH) <= (OUTPUT_WIDTH - INPUT_WIDTH - 1 downto 0 => ins(INPUT_WIDTH - 1));
  outs(INPUT_WIDTH - 1 downto 0)            <= ins;
  outs_valid                                <= ins_valid;
  ins_ready                                 <= outs_ready;
end architecture;
