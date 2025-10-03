## bifgen
generate bif files for roku devices in python without piping out to shell commands.

### usage:
`bifgen [-h] [-i N] [-O N] [-o FILE] [--sd] [-s] [-j N] [--hwaccel HWACCEL] [--preset PRESET] sourcevid`

#### positional arguments
| argument               |                                               description |
| --:                    |                                                       :-- |
| `sourcevid`            |                                     video file to process |

#### optional arguments
| argument               |                                               description |
| --:                    |                                                       :-- |
| `-h`, `--help`         |                           show this help message and exit |
| `-i N`, `--interval N` |        interval between images in seconds (10 by default) |
| `-O N`, `--offset N`   |           offset to first image in seconds (0 by default) |
| `-o FILE`, `--out FILE`|          destination path/file where result will be saved |
| `--sd`                 |               resulting bif file will be sd instead of hd |
| `-s`, `--silent`       | do not print progress or diagnostic information to stdout |
| `-j N`, `--jobs N`     |    Number of parallel jobs to run (defaults to core count) |
| `--hwaccel HWACCEL`    |   Hardware acceleration to use (default: auto). Options: auto, cuda, videotoolbox, none |
| `--preset PRESET`      |   Speed vs quality preset (default: medium). Options: fast, medium, quality |


