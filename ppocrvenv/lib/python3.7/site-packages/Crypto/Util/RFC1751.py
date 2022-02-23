# rfc1751.py : Converts between 128-bit strings and a human-readable
# sequence of words, as defined in RFC1751: "A Convention for
# Human-Readable 128-bit Keys", by Daniel L. McDonald.
#
# Part of the Python Cryptography Toolkit
#
# Written by Andrew M. Kuchling and others
#
# ===================================================================
# The contents of this file are dedicated to the public domain.  To
# the extent that dedication to the public domain is not available,
# everyone is granted a worldwide, perpetual, royalty-free,
# non-exclusive license to exercise all rights associated with the
# contents of this file for any purpose whatsoever.
# No rights are reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================

from __future__ import print_function

import binascii

from Crypto.Util.py3compat import bord, bchr

binary = {0: '0000', 1: '0001', 2: '0010', 3: '0011', 4: '0100', 5: '0101',
          6: '0110', 7: '0111', 8: '1000', 9: '1001', 10: '1010', 11: '1011',
          12: '1100', 13: '1101', 14: '1110', 15: '1111'}


def _key2bin(s):
    "Convert a key into a string of binary digits"
    kl = map(lambda x: bord(x), s)
    kl = map(lambda x: binary[x >> 4] + binary[x & 15], kl)
    return ''.join(kl)


def _extract(key, start, length):
    """Extract a bitstring(2.x)/bytestring(2.x) from a string of binary digits, and return its
    numeric value."""

    result = 0
    for y in key[start:start+length]:
        result = result * 2 + ord(y) - 48
    return result


def key_to_english(key):
    """Transform an arbitrary key into a string containing English words.

    Example::

        >>> from Crypto.Util.RFC1751 import key_to_english
        >>> key_to_english(b'66666666')
        'RAM LOIS GOAD CREW CARE HIT'

    Args:
      key (byte string):
        The key to convert. Its length must be a multiple of 8.
    Return:
      A string of English words.
    """

    if len(key) % 8 != 0:
        raise ValueError('The length of the key must be a multiple of 8.')

    english = ''
    for index in range(0, len(key), 8):  # Loop over 8-byte subkeys
        subkey = key[index:index + 8]
        # Compute the parity of the key
        skbin = _key2bin(subkey)
        p = 0
        for i in range(0, 64, 2):
            p = p + _extract(skbin, i, 2)
        # Append parity bits to the subkey
        skbin = _key2bin(subkey + bchr((p << 6) & 255))
        for i in range(0, 64, 11):
            english = english + wordlist[_extract(skbin, i, 11)] + ' '

    return english.strip()


def english_to_key(s):
    """Transform a string into a corresponding key.

    Example::

        >>> from Crypto.Util.RFC1751 import english_to_key
        >>> english_to_key('RAM LOIS GOAD CREW CARE HIT')
        b'66666666'

    Args:
      s (string): the string with the words separated by whitespace;
                  the number of words must be a multiple of 6.
    Return:
      A byte string.
    """

    L = s.upper().split()
    key = b''
    for index in range(0, len(L), 6):
        sublist = L[index:index + 6]
        char = 9 * [0]
        bits = 0
        for i in sublist:
            index = wordlist.index(i)
            shift = (8 - (bits + 11) % 8) % 8
            y = index << shift
            cl, cc, cr = (y >> 16), (y >> 8) & 0xff, y & 0xff
            if (shift > 5):
                char[bits >> 3] = char[bits >> 3] | cl
                char[(bits >> 3) + 1] = char[(bits >> 3) + 1] | cc
                char[(bits >> 3) + 2] = char[(bits >> 3) + 2] | cr
            elif shift > -3:
                char[bits >> 3] = char[bits >> 3] | cc
                char[(bits >> 3) + 1] = char[(bits >> 3) + 1] | cr
            else:
                char[bits >> 3] = char[bits >> 3] | cr
            bits = bits + 11

        subkey = b''
        for y in char:
            subkey = subkey + bchr(y)

        # Check the parity of the resulting key
        skbin = _key2bin(subkey)
        p = 0
        for i in range(0, 64, 2):
            p = p + _extract(skbin, i, 2)
        if (p & 3) != _extract(skbin, 64, 2):
            raise ValueError("Parity error in resulting key")
        key = key + subkey[0:8]
    return key


wordlist = [
   "A", "ABE", "ACE", "ACT", "AD", "ADA", "ADD",
   "AGO", "AID", "AIM", "AIR", "ALL", "ALP", "AM", "AMY", "AN", "ANA",
   "AND", "ANN", "ANT", "ANY", "APE", "APS", "APT", "ARC", "ARE", "ARK",
   "ARM", "ART", "AS", "ASH", "ASK", "AT", "ATE", "AUG", "AUK", "AVE",
   "AWE", "AWK", "AWL", "AWN", "AX", "AYE", "BAD", "BAG", "BAH", "BAM",
   "BAN", "BAR", "BAT", "BAY", "BE", "BED", "BEE", "BEG", "BEN", "BET",
   "BEY", "BIB", "BID", "BIG", "BIN", "BIT", "BOB", "BOG", "BON", "BOO",
   "BOP", "BOW", "BOY", "BUB", "BUD", "BUG", "BUM", "BUN", "BUS", "BUT",
   "BUY", "BY", "BYE", "CAB", "CAL", "CAM", "CAN", "CAP", "CAR", "CAT",
   "CAW", "COD", "COG", "COL", "CON", "COO", "COP", "COT", "COW", "COY",
   "CRY", "CUB", "CUE", "CUP", "CUR", "CUT", "DAB", "DAD", "DAM", "DAN",
   "DAR", "DAY", "DEE", "DEL", "DEN", "DES", "DEW", "DID", "DIE", "DIG",
   "DIN", "DIP", "DO", "DOE", "DOG", "DON", "DOT", "DOW", "DRY", "DUB",
   "DUD", "DUE", "DUG", "DUN", "EAR", "EAT", "ED", "EEL", "EGG", "EGO",
   "ELI", "ELK", "ELM", "ELY", "EM", "END", "EST", "ETC", "EVA", "EVE",
   "EWE", "EYE", "FAD", "FAN", "FAR", "FAT", "FAY", "FED", "FEE", "FEW",
   "FIB", "FIG", "FIN", "FIR", "FIT", "FLO", "FLY", "FOE", "FOG", "FOR",
   "FRY", "FUM", "FUN", "FUR", "GAB", "GAD", "GAG", "GAL", "GAM", "GAP",
   "GAS", "GAY", "GEE", "GEL", "GEM", "GET", "GIG", "GIL", "GIN", "GO",
   "GOT", "GUM", "GUN", "GUS", "GUT", "GUY", "GYM", "GYP", "HA", "HAD",
   "HAL", "HAM", "HAN", "HAP", "HAS", "HAT", "HAW", "HAY", "HE", "HEM",
   "HEN", "HER", "HEW", "HEY", "HI", "HID", "HIM", "HIP", "HIS", "HIT",
   "HO", "HOB", "HOC", "HOE", "HOG", "HOP", "HOT", "HOW", "HUB", "HUE",
   "HUG", "HUH", "HUM", "HUT", "I", "ICY", "IDA", "IF", "IKE", "ILL",
   "INK", "INN", "IO", "ION", "IQ", "IRA", "IRE", "IRK", "IS", "IT",
   "ITS", "IVY", "JAB", "JAG", "JAM", "JAN", "JAR", "JAW", "JAY", "JET",
   "JIG", "JIM", "JO", "JOB", "JOE", "JOG", "JOT", "JOY", "JUG", "JUT",
   "KAY", "KEG", "KEN", "KEY", "KID", "KIM", "KIN", "KIT", "LA", "LAB",
   "LAC", "LAD", "LAG", "LAM", "LAP", "LAW", "LAY", "LEA", "LED", "LEE",
   "LEG", "LEN", "LEO", "LET", "LEW", "LID", "LIE", "LIN", "LIP", "LIT",
   "LO", "LOB", "LOG", "LOP", "LOS", "LOT", "LOU", "LOW", "LOY", "LUG",
   "LYE", "MA", "MAC", "MAD", "MAE", "MAN", "MAO", "MAP", "MAT", "MAW",
   "MAY", "ME", "MEG", "MEL", "MEN", "MET", "MEW", "MID", "MIN", "MIT",
   "MOB", "MOD", "MOE", "MOO", "MOP", "MOS", "MOT", "MOW", "MUD", "MUG",
   "MUM", "MY", "NAB", "NAG", "NAN", "NAP", "NAT", "NAY", "NE", "NED",
   "NEE", "NET", "NEW", "NIB", "NIL", "NIP", "NIT", "NO", "NOB", "NOD",
   "NON", "NOR", "NOT", "NOV", "NOW", "NU", "NUN", "NUT", "O", "OAF",
   "OAK", "OAR", "OAT", "ODD", "ODE", "OF", "OFF", "OFT", "OH", "OIL",
   "OK", "OLD", "ON", "ONE", "OR", "ORB", "ORE", "ORR", "OS", "OTT",
   "OUR", "OUT", "OVA", "OW", "OWE", "OWL", "OWN", "OX", "PA", "PAD",
   "PAL", "PAM", "PAN", "PAP", "PAR", "PAT", "PAW", "PAY", "PEA", "PEG",
   "PEN", "PEP", "PER", "PET", "PEW", "PHI", "PI", "PIE", "PIN", "PIT",
   "PLY", "PO", "POD", "POE", "POP", "POT", "POW", "PRO", "PRY", "PUB",
   "PUG", "PUN", "PUP", "PUT", "QUO", "RAG", "RAM", "RAN", "RAP", "RAT",
   "RAW", "RAY", "REB", "RED", "REP", "RET", "RIB", "RID", "RIG", "RIM",
   "RIO", "RIP", "ROB", "ROD", "ROE", "RON", "ROT", "ROW", "ROY", "RUB",
   "RUE", "RUG", "RUM", "RUN", "RYE", "SAC", "SAD", "SAG", "SAL", "SAM",
   "SAN", "SAP", "SAT", "SAW", "SAY", "SEA", "SEC", "SEE", "SEN", "SET",
   "SEW", "SHE", "SHY", "SIN", "SIP", "SIR", "SIS", "SIT", "SKI", "SKY",
   "SLY", "SO", "SOB", "SOD", "SON", "SOP", "SOW", "SOY", "SPA", "SPY",
   "SUB", "SUD", "SUE", "SUM", "SUN", "SUP", "TAB", "TAD", "TAG", "TAN",
   "TAP", "TAR", "TEA", "TED", "TEE", "TEN", "THE", "THY", "TIC", "TIE",
   "TIM", "TIN", "TIP", "TO", "TOE", "TOG", "TOM", "TON", "TOO", "TOP",
   "TOW", "TOY", "TRY", "TUB", "TUG", "TUM", "TUN", "TWO", "UN", "UP",
   "US", "USE", "VAN", "VAT", "VET", "VIE", "WAD", "WAG", "WAR", "WAS",
   "WAY", "WE", "WEB", "WED", "WEE", "WET", "WHO", "WHY", "WIN", "WIT",
   "WOK", "WON", "WOO", "WOW", "WRY", "WU", "YAM", "YAP", "YAW", "YE",
   "YEA", "YES", "YET", "YOU", "ABED", "ABEL", "ABET", "ABLE", "ABUT",
   "ACHE", "ACID", "ACME", "ACRE", "ACTA", "ACTS", "ADAM", "ADDS",
   "ADEN", "AFAR", "AFRO", "AGEE", "AHEM", "AHOY", "AIDA", "AIDE",
   "AIDS", "AIRY", "AJAR", "AKIN", "ALAN", "ALEC", "ALGA", "ALIA",
   "ALLY", "ALMA", "ALOE", "ALSO", "ALTO", "ALUM", "ALVA", "AMEN",
   "AMES", "AMID", "AMMO", "AMOK", "AMOS", "AMRA", "ANDY", "ANEW",
   "ANNA", "ANNE", "ANTE", "ANTI", "AQUA", "ARAB", "ARCH", "AREA",
   "ARGO", "ARID", "ARMY", "ARTS", "ARTY", "ASIA", "ASKS", "ATOM",
   "AUNT", "AURA", "AUTO", "AVER", "AVID", "AVIS", "AVON", "AVOW",
   "AWAY", "AWRY", "BABE", "BABY", "BACH", "BACK", "BADE", "BAIL",
   "BAIT", "BAKE", "BALD", "BALE", "BALI", "BALK", "BALL", "BALM",
   "BAND", "BANE", "BANG", "BANK", "BARB", "BARD", "BARE", "BARK",
   "BARN", "BARR", "BASE", "BASH", "BASK", "BASS", "BATE", "BATH",
   "BAWD", "BAWL", "BEAD", "BEAK", "BEAM", "BEAN", "BEAR", "BEAT",
   "BEAU", "BECK", "BEEF", "BEEN", "BEER",
   "BEET", "BELA", "BELL", "BELT", "BEND", "BENT", "BERG", "BERN",
   "BERT", "BESS", "BEST", "BETA", "BETH", "BHOY", "BIAS", "BIDE",
   "BIEN", "BILE", "BILK", "BILL", "BIND", "BING", "BIRD", "BITE",
   "BITS", "BLAB", "BLAT", "BLED", "BLEW", "BLOB", "BLOC", "BLOT",
   "BLOW", "BLUE", "BLUM", "BLUR", "BOAR", "BOAT", "BOCA", "BOCK",
   "BODE", "BODY", "BOGY", "BOHR", "BOIL", "BOLD", "BOLO", "BOLT",
   "BOMB", "BONA", "BOND", "BONE", "BONG", "BONN", "BONY", "BOOK",
   "BOOM", "BOON", "BOOT", "BORE", "BORG", "BORN", "BOSE", "BOSS",
   "BOTH", "BOUT", "BOWL", "BOYD", "BRAD", "BRAE", "BRAG", "BRAN",
   "BRAY", "BRED", "BREW", "BRIG", "BRIM", "BROW", "BUCK", "BUDD",
   "BUFF", "BULB", "BULK", "BULL", "BUNK", "BUNT", "BUOY", "BURG",
   "BURL", "BURN", "BURR", "BURT", "BURY", "BUSH", "BUSS", "BUST",
   "BUSY", "BYTE", "CADY", "CAFE", "CAGE", "CAIN", "CAKE", "CALF",
   "CALL", "CALM", "CAME", "CANE", "CANT", "CARD", "CARE", "CARL",
   "CARR", "CART", "CASE", "CASH", "CASK", "CAST", "CAVE", "CEIL",
   "CELL", "CENT", "CERN", "CHAD", "CHAR", "CHAT", "CHAW", "CHEF",
   "CHEN", "CHEW", "CHIC", "CHIN", "CHOU", "CHOW", "CHUB", "CHUG",
   "CHUM", "CITE", "CITY", "CLAD", "CLAM", "CLAN", "CLAW", "CLAY",
   "CLOD", "CLOG", "CLOT", "CLUB", "CLUE", "COAL", "COAT", "COCA",
   "COCK", "COCO", "CODA", "CODE", "CODY", "COED", "COIL", "COIN",
   "COKE", "COLA", "COLD", "COLT", "COMA", "COMB", "COME", "COOK",
   "COOL", "COON", "COOT", "CORD", "CORE", "CORK", "CORN", "COST",
   "COVE", "COWL", "CRAB", "CRAG", "CRAM", "CRAY", "CREW", "CRIB",
   "CROW", "CRUD", "CUBA", "CUBE", "CUFF", "CULL", "CULT", "CUNY",
   "CURB", "CURD", "CURE", "CURL", "CURT", "CUTS", "DADE", "DALE",
   "DAME", "DANA", "DANE", "DANG", "DANK", "DARE", "DARK", "DARN",
   "DART", "DASH", "DATA", "DATE", "DAVE", "DAVY", "DAWN", "DAYS",
   "DEAD", "DEAF", "DEAL", "DEAN", "DEAR", "DEBT", "DECK", "DEED",
   "DEEM", "DEER", "DEFT", "DEFY", "DELL", "DENT", "DENY", "DESK",
   "DIAL", "DICE", "DIED", "DIET", "DIME", "DINE", "DING", "DINT",
   "DIRE", "DIRT", "DISC", "DISH", "DISK", "DIVE", "DOCK", "DOES",
   "DOLE", "DOLL", "DOLT", "DOME", "DONE", "DOOM", "DOOR", "DORA",
   "DOSE", "DOTE", "DOUG", "DOUR", "DOVE", "DOWN", "DRAB", "DRAG",
   "DRAM", "DRAW", "DREW", "DRUB", "DRUG", "DRUM", "DUAL", "DUCK",
   "DUCT", "DUEL", "DUET", "DUKE", "DULL", "DUMB", "DUNE", "DUNK",
   "DUSK", "DUST", "DUTY", "EACH", "EARL", "EARN", "EASE", "EAST",
   "EASY", "EBEN", "ECHO", "EDDY", "EDEN", "EDGE", "EDGY", "EDIT",
   "EDNA", "EGAN", "ELAN", "ELBA", "ELLA", "ELSE", "EMIL", "EMIT",
   "EMMA", "ENDS", "ERIC", "EROS", "EVEN", "EVER", "EVIL", "EYED",
   "FACE", "FACT", "FADE", "FAIL", "FAIN", "FAIR", "FAKE", "FALL",
   "FAME", "FANG", "FARM", "FAST", "FATE", "FAWN", "FEAR", "FEAT",
   "FEED", "FEEL", "FEET", "FELL", "FELT", "FEND", "FERN", "FEST",
   "FEUD", "FIEF", "FIGS", "FILE", "FILL", "FILM", "FIND", "FINE",
   "FINK", "FIRE", "FIRM", "FISH", "FISK", "FIST", "FITS", "FIVE",
   "FLAG", "FLAK", "FLAM", "FLAT", "FLAW", "FLEA", "FLED", "FLEW",
   "FLIT", "FLOC", "FLOG", "FLOW", "FLUB", "FLUE", "FOAL", "FOAM",
   "FOGY", "FOIL", "FOLD", "FOLK", "FOND", "FONT", "FOOD", "FOOL",
   "FOOT", "FORD", "FORE", "FORK", "FORM", "FORT", "FOSS", "FOUL",
   "FOUR", "FOWL", "FRAU", "FRAY", "FRED", "FREE", "FRET", "FREY",
   "FROG", "FROM", "FUEL", "FULL", "FUME", "FUND", "FUNK", "FURY",
   "FUSE", "FUSS", "GAFF", "GAGE", "GAIL", "GAIN", "GAIT", "GALA",
   "GALE", "GALL", "GALT", "GAME", "GANG", "GARB", "GARY", "GASH",
   "GATE", "GAUL", "GAUR", "GAVE", "GAWK", "GEAR", "GELD", "GENE",
   "GENT", "GERM", "GETS", "GIBE", "GIFT", "GILD", "GILL", "GILT",
   "GINA", "GIRD", "GIRL", "GIST", "GIVE", "GLAD", "GLEE", "GLEN",
   "GLIB", "GLOB", "GLOM", "GLOW", "GLUE", "GLUM", "GLUT", "GOAD",
   "GOAL", "GOAT", "GOER", "GOES", "GOLD", "GOLF", "GONE", "GONG",
   "GOOD", "GOOF", "GORE", "GORY", "GOSH", "GOUT", "GOWN", "GRAB",
   "GRAD", "GRAY", "GREG", "GREW", "GREY", "GRID", "GRIM", "GRIN",
   "GRIT", "GROW", "GRUB", "GULF", "GULL", "GUNK", "GURU", "GUSH",
   "GUST", "GWEN", "GWYN", "HAAG", "HAAS", "HACK", "HAIL", "HAIR",
   "HALE", "HALF", "HALL", "HALO", "HALT", "HAND", "HANG", "HANK",
   "HANS", "HARD", "HARK", "HARM", "HART", "HASH", "HAST", "HATE",
   "HATH", "HAUL", "HAVE", "HAWK", "HAYS", "HEAD", "HEAL", "HEAR",
   "HEAT", "HEBE", "HECK", "HEED", "HEEL", "HEFT", "HELD", "HELL",
   "HELM", "HERB", "HERD", "HERE", "HERO", "HERS", "HESS", "HEWN",
   "HICK", "HIDE", "HIGH", "HIKE", "HILL", "HILT", "HIND", "HINT",
   "HIRE", "HISS", "HIVE", "HOBO", "HOCK", "HOFF", "HOLD", "HOLE",
   "HOLM", "HOLT", "HOME", "HONE", "HONK", "HOOD", "HOOF", "HOOK",
   "HOOT", "HORN", "HOSE", "HOST", "HOUR", "HOVE", "HOWE", "HOWL",
   "HOYT", "HUCK", "HUED", "HUFF", "HUGE", "HUGH", "HUGO", "HULK",
   "HULL", "HUNK", "HUNT", "HURD", "HURL", "HURT", "HUSH", "HYDE",
   "HYMN", "IBIS", "ICON", "IDEA", "IDLE", "IFFY", "INCA", "INCH",
   "INTO", "IONS", "IOTA", "IOWA", "IRIS", "IRMA", "IRON", "ISLE",
   "ITCH", "ITEM", "IVAN", "JACK", "JADE", "JAIL", "JAKE", "JANE",
   "JAVA", "JEAN", "JEFF", "JERK", "JESS", "JEST", "JIBE", "JILL",
   "JILT", "JIVE", "JOAN", "JOBS", "JOCK", "JOEL", "JOEY", "JOHN",
   "JOIN", "JOKE", "JOLT", "JOVE", "JUDD", "JUDE", "JUDO", "JUDY",
   "JUJU", "JUKE", "JULY", "JUNE", "JUNK", "JUNO", "JURY", "JUST",
   "JUTE", "KAHN", "KALE", "KANE", "KANT", "KARL", "KATE", "KEEL",
   "KEEN", "KENO", "KENT", "KERN", "KERR", "KEYS", "KICK", "KILL",
   "KIND", "KING", "KIRK", "KISS", "KITE", "KLAN", "KNEE", "KNEW",
   "KNIT", "KNOB", "KNOT", "KNOW", "KOCH", "KONG", "KUDO", "KURD",
   "KURT", "KYLE", "LACE", "LACK", "LACY", "LADY", "LAID", "LAIN",
   "LAIR", "LAKE", "LAMB", "LAME", "LAND", "LANE", "LANG", "LARD",
   "LARK", "LASS", "LAST", "LATE", "LAUD", "LAVA", "LAWN", "LAWS",
   "LAYS", "LEAD", "LEAF", "LEAK", "LEAN", "LEAR", "LEEK", "LEER",
   "LEFT", "LEND", "LENS", "LENT", "LEON", "LESK", "LESS", "LEST",
   "LETS", "LIAR", "LICE", "LICK", "LIED", "LIEN", "LIES", "LIEU",
   "LIFE", "LIFT", "LIKE", "LILA", "LILT", "LILY", "LIMA", "LIMB",
   "LIME", "LIND", "LINE", "LINK", "LINT", "LION", "LISA", "LIST",
   "LIVE", "LOAD", "LOAF", "LOAM", "LOAN", "LOCK", "LOFT", "LOGE",
   "LOIS", "LOLA", "LONE", "LONG", "LOOK", "LOON", "LOOT", "LORD",
   "LORE", "LOSE", "LOSS", "LOST", "LOUD", "LOVE", "LOWE", "LUCK",
   "LUCY", "LUGE", "LUKE", "LULU", "LUND", "LUNG", "LURA", "LURE",
   "LURK", "LUSH", "LUST", "LYLE", "LYNN", "LYON", "LYRA", "MACE",
   "MADE", "MAGI", "MAID", "MAIL", "MAIN", "MAKE", "MALE", "MALI",
   "MALL", "MALT", "MANA", "MANN", "MANY", "MARC", "MARE", "MARK",
   "MARS", "MART", "MARY", "MASH", "MASK", "MASS", "MAST", "MATE",
   "MATH", "MAUL", "MAYO", "MEAD", "MEAL", "MEAN", "MEAT", "MEEK",
   "MEET", "MELD", "MELT", "MEMO", "MEND", "MENU", "MERT", "MESH",
   "MESS", "MICE", "MIKE", "MILD", "MILE", "MILK", "MILL", "MILT",
   "MIMI", "MIND", "MINE", "MINI", "MINK", "MINT", "MIRE", "MISS",
   "MIST", "MITE", "MITT", "MOAN", "MOAT", "MOCK", "MODE", "MOLD",
   "MOLE", "MOLL", "MOLT", "MONA", "MONK", "MONT", "MOOD", "MOON",
   "MOOR", "MOOT", "MORE", "MORN", "MORT", "MOSS", "MOST", "MOTH",
   "MOVE", "MUCH", "MUCK", "MUDD", "MUFF", "MULE", "MULL", "MURK",
   "MUSH", "MUST", "MUTE", "MUTT", "MYRA", "MYTH", "NAGY", "NAIL",
   "NAIR", "NAME", "NARY", "NASH", "NAVE", "NAVY", "NEAL", "NEAR",
   "NEAT", "NECK", "NEED", "NEIL", "NELL", "NEON", "NERO", "NESS",
   "NEST", "NEWS", "NEWT", "NIBS", "NICE", "NICK", "NILE", "NINA",
   "NINE", "NOAH", "NODE", "NOEL", "NOLL", "NONE", "NOOK", "NOON",
   "NORM", "NOSE", "NOTE", "NOUN", "NOVA", "NUDE", "NULL", "NUMB",
   "OATH", "OBEY", "OBOE", "ODIN", "OHIO", "OILY", "OINT", "OKAY",
   "OLAF", "OLDY", "OLGA", "OLIN", "OMAN", "OMEN", "OMIT", "ONCE",
   "ONES", "ONLY", "ONTO", "ONUS", "ORAL", "ORGY", "OSLO", "OTIS",
   "OTTO", "OUCH", "OUST", "OUTS", "OVAL", "OVEN", "OVER", "OWLY",
   "OWNS", "QUAD", "QUIT", "QUOD", "RACE", "RACK", "RACY", "RAFT",
   "RAGE", "RAID", "RAIL", "RAIN", "RAKE", "RANK", "RANT", "RARE",
   "RASH", "RATE", "RAVE", "RAYS", "READ", "REAL", "REAM", "REAR",
   "RECK", "REED", "REEF", "REEK", "REEL", "REID", "REIN", "RENA",
   "REND", "RENT", "REST", "RICE", "RICH", "RICK", "RIDE", "RIFT",
   "RILL", "RIME", "RING", "RINK", "RISE", "RISK", "RITE", "ROAD",
   "ROAM", "ROAR", "ROBE", "ROCK", "RODE", "ROIL", "ROLL", "ROME",
   "ROOD", "ROOF", "ROOK", "ROOM", "ROOT", "ROSA", "ROSE", "ROSS",
   "ROSY", "ROTH", "ROUT", "ROVE", "ROWE", "ROWS", "RUBE", "RUBY",
   "RUDE", "RUDY", "RUIN", "RULE", "RUNG", "RUNS", "RUNT", "RUSE",
   "RUSH", "RUSK", "RUSS", "RUST", "RUTH", "SACK", "SAFE", "SAGE",
   "SAID", "SAIL", "SALE", "SALK", "SALT", "SAME", "SAND", "SANE",
   "SANG", "SANK", "SARA", "SAUL", "SAVE", "SAYS", "SCAN", "SCAR",
   "SCAT", "SCOT", "SEAL", "SEAM", "SEAR", "SEAT", "SEED", "SEEK",
   "SEEM", "SEEN", "SEES", "SELF", "SELL", "SEND", "SENT", "SETS",
   "SEWN", "SHAG", "SHAM", "SHAW", "SHAY", "SHED", "SHIM", "SHIN",
   "SHOD", "SHOE", "SHOT", "SHOW", "SHUN", "SHUT", "SICK", "SIDE",
   "SIFT", "SIGH", "SIGN", "SILK", "SILL", "SILO", "SILT", "SINE",
   "SING", "SINK", "SIRE", "SITE", "SITS", "SITU", "SKAT", "SKEW",
   "SKID", "SKIM", "SKIN", "SKIT", "SLAB", "SLAM", "SLAT", "SLAY",
   "SLED", "SLEW", "SLID", "SLIM", "SLIT", "SLOB", "SLOG", "SLOT",
   "SLOW", "SLUG", "SLUM", "SLUR", "SMOG", "SMUG", "SNAG", "SNOB",
   "SNOW", "SNUB", "SNUG", "SOAK", "SOAR", "SOCK", "SODA", "SOFA",
   "SOFT", "SOIL", "SOLD", "SOME", "SONG", "SOON", "SOOT", "SORE",
   "SORT", "SOUL", "SOUR", "SOWN", "STAB", "STAG", "STAN", "STAR",
   "STAY", "STEM", "STEW", "STIR", "STOW", "STUB", "STUN", "SUCH",
   "SUDS", "SUIT", "SULK", "SUMS", "SUNG", "SUNK", "SURE", "SURF",
   "SWAB", "SWAG", "SWAM", "SWAN", "SWAT", "SWAY", "SWIM", "SWUM",
   "TACK", "TACT", "TAIL", "TAKE", "TALE", "TALK", "TALL", "TANK",
   "TASK", "TATE", "TAUT", "TEAL", "TEAM", "TEAR", "TECH", "TEEM",
   "TEEN", "TEET", "TELL", "TEND", "TENT", "TERM", "TERN", "TESS",
   "TEST", "THAN", "THAT", "THEE", "THEM", "THEN", "THEY", "THIN",
   "THIS", "THUD", "THUG", "TICK", "TIDE", "TIDY", "TIED", "TIER",
   "TILE", "TILL", "TILT", "TIME", "TINA", "TINE", "TINT", "TINY",
   "TIRE", "TOAD", "TOGO", "TOIL", "TOLD", "TOLL", "TONE", "TONG",
   "TONY", "TOOK", "TOOL", "TOOT", "TORE", "TORN", "TOTE", "TOUR",
   "TOUT", "TOWN", "TRAG", "TRAM", "TRAY", "TREE", "TREK", "TRIG",
   "TRIM", "TRIO", "TROD", "TROT", "TROY", "TRUE", "TUBA", "TUBE",
   "TUCK", "TUFT", "TUNA", "TUNE", "TUNG", "TURF", "TURN", "TUSK",
   "TWIG", "TWIN", "TWIT", "ULAN", "UNIT", "URGE", "USED", "USER",
   "USES", "UTAH", "VAIL", "VAIN", "VALE", "VARY", "VASE", "VAST",
   "VEAL", "VEDA", "VEIL", "VEIN", "VEND", "VENT", "VERB", "VERY",
   "VETO", "VICE", "VIEW", "VINE", "VISE", "VOID", "VOLT", "VOTE",
   "WACK", "WADE", "WAGE", "WAIL", "WAIT", "WAKE", "WALE", "WALK",
   "WALL", "WALT", "WAND", "WANE", "WANG", "WANT", "WARD", "WARM",
   "WARN", "WART", "WASH", "WAST", "WATS", "WATT", "WAVE", "WAVY",
   "WAYS", "WEAK", "WEAL", "WEAN", "WEAR", "WEED", "WEEK", "WEIR",
   "WELD", "WELL", "WELT", "WENT", "WERE", "WERT", "WEST", "WHAM",
   "WHAT", "WHEE", "WHEN", "WHET", "WHOA", "WHOM", "WICK", "WIFE",
   "WILD", "WILL", "WIND", "WINE", "WING", "WINK", "WINO", "WIRE",
   "WISE", "WISH", "WITH", "WOLF", "WONT", "WOOD", "WOOL", "WORD",
   "WORE", "WORK", "WORM", "WORN", "WOVE", "WRIT", "WYNN", "YALE",
   "YANG", "YANK", "YARD", "YARN", "YAWL", "YAWN", "YEAH", "YEAR",
   "YELL", "YOGA", "YOKE" ]
