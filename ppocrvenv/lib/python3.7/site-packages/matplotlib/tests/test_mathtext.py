import io
from pathlib import Path
import re
import shlex
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
from matplotlib import _api, mathtext


# If test is removed, use None as placeholder
math_tests = [
    r'$a+b+\dot s+\dot{s}+\ldots$',
    r'$x \doteq y$',
    r'\$100.00 $\alpha \_$',
    r'$\frac{\$100.00}{y}$',
    r'$x   y$',
    r'$x+y\ x=y\ x<y\ x:y\ x,y\ x@y$',
    r'$100\%y\ x*y\ x/y x\$y$',
    r'$x\leftarrow y\ x\forall y\ x-y$',
    r'$x \sf x \bf x {\cal X} \rm x$',
    r'$x\ x\,x\;x\quad x\qquad x\!x\hspace{ 0.5 }y$',
    r'$\{ \rm braces \}$',
    r'$\left[\left\lfloor\frac{5}{\frac{\left(3\right)}{4}} y\right)\right]$',
    r'$\left(x\right)$',
    r'$\sin(x)$',
    r'$x_2$',
    r'$x^2$',
    r'$x^2_y$',
    r'$x_y^2$',
    (r'$\sum _{\genfrac{}{}{0}{}{0\leq i\leq m}{0<j<n}}f\left(i,j\right)'
     r'\mathcal{R}\prod_{i=\alpha_{i+1}}^\infty a_i \sin(2 \pi f x_i)'
     r"\sqrt[2]{\prod^\frac{x}{2\pi^2}_\infty}$"),
    r'$x = \frac{x+\frac{5}{2}}{\frac{y+3}{8}}$',
    r'$dz/dt = \gamma x^2 + {\rm sin}(2\pi y+\phi)$',
    r'Foo: $\alpha_{i+1}^j = {\rm sin}(2\pi f_j t_i) e^{-5 t_i/\tau}$',
    None,
    r'Variable $i$ is good',
    r'$\Delta_i^j$',
    r'$\Delta^j_{i+1}$',
    r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{\imath}\tilde{n}\vec{q}$',
    r"$\arccos((x^i))$",
    r"$\gamma = \frac{x=\frac{6}{8}}{y} \delta$",
    r'$\limsup_{x\to\infty}$',
    r'$\oint^\infty_0$',
    r"$f'\quad f'''(x)\quad ''/\mathrm{yr}$",
    r'$\frac{x_2888}{y}$',
    r"$\sqrt[3]{\frac{X_2}{Y}}=5$",
    None,
    r"$\sqrt[3]{x}=5$",
    r'$\frac{X}{\frac{X}{Y}}$',
    r"$W^{3\beta}_{\delta_1 \rho_1 \sigma_2} = U^{3\beta}_{\delta_1 \rho_1} + \frac{1}{8 \pi 2} \int^{\alpha_2}_{\alpha_2} d \alpha^\prime_2 \left[\frac{ U^{2\beta}_{\delta_1 \rho_1} - \alpha^\prime_2U^{1\beta}_{\rho_1 \sigma_2} }{U^{0\beta}_{\rho_1 \sigma_2}}\right]$",
    r'$\mathcal{H} = \int d \tau \left(\epsilon E^2 + \mu H^2\right)$',
    r'$\widehat{abc}\widetilde{def}$',
    '$\\Gamma \\Delta \\Theta \\Lambda \\Xi \\Pi \\Sigma \\Upsilon \\Phi \\Psi \\Omega$',
    '$\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta \\theta \\iota \\lambda \\mu \\nu \\xi \\pi \\kappa \\rho \\sigma \\tau \\upsilon \\phi \\chi \\psi$',

    # The examples prefixed by 'mmltt' are from the MathML torture test here:
    # https://developer.mozilla.org/en-US/docs/Mozilla/MathML_Project/MathML_Torture_Test
    r'${x}^{2}{y}^{2}$',
    r'${}_{2}F_{3}$',
    r'$\frac{x+{y}^{2}}{k+1}$',
    r'$x+{y}^{\frac{2}{k+1}}$',
    r'$\frac{a}{b/2}$',
    r'${a}_{0}+\frac{1}{{a}_{1}+\frac{1}{{a}_{2}+\frac{1}{{a}_{3}+\frac{1}{{a}_{4}}}}}$',
    r'${a}_{0}+\frac{1}{{a}_{1}+\frac{1}{{a}_{2}+\frac{1}{{a}_{3}+\frac{1}{{a}_{4}}}}}$',
    r'$\binom{n}{k/2}$',
    r'$\binom{p}{2}{x}^{2}{y}^{p-2}-\frac{1}{1-x}\frac{1}{1-{x}^{2}}$',
    r'${x}^{2y}$',
    r'$\sum _{i=1}^{p}\sum _{j=1}^{q}\sum _{k=1}^{r}{a}_{ij}{b}_{jk}{c}_{ki}$',
    r'$\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+\sqrt{1+x}}}}}}}$',
    r'$\left(\frac{{\partial }^{2}}{\partial {x}^{2}}+\frac{{\partial }^{2}}{\partial {y}^{2}}\right){|\varphi \left(x+iy\right)|}^{2}=0$',
    r'${2}^{{2}^{{2}^{x}}}$',
    r'${\int }_{1}^{x}\frac{\mathrm{dt}}{t}$',
    r'$\int {\int }_{D}\mathrm{dx} \mathrm{dy}$',
    # mathtex doesn't support array
    # 'mmltt18'    : r'$f\left(x\right)=\left\{\begin{array}{cc}\hfill 1/3\hfill & \text{if_}0\le x\le 1;\hfill \\ \hfill 2/3\hfill & \hfill \text{if_}3\le x\le 4;\hfill \\ \hfill 0\hfill & \text{elsewhere.}\hfill \end{array}$',
    # mathtex doesn't support stackrel
    # 'mmltt19'    : r'$\stackrel{\stackrel{k\text{times}}{\ufe37}}{x+...+x}$',
    r'${y}_{{x}^{2}}$',
    # mathtex doesn't support the "\text" command
    # 'mmltt21'    : r'$\sum _{p\text{\prime}}f\left(p\right)={\int }_{t>1}f\left(t\right) d\pi \left(t\right)$',
    # mathtex doesn't support array
    # 'mmltt23'    : r'$\left(\begin{array}{cc}\hfill \left(\begin{array}{cc}\hfill a\hfill & \hfill b\hfill \\ \hfill c\hfill & \hfill d\hfill \end{array}\right)\hfill & \hfill \left(\begin{array}{cc}\hfill e\hfill & \hfill f\hfill \\ \hfill g\hfill & \hfill h\hfill \end{array}\right)\hfill \\ \hfill 0\hfill & \hfill \left(\begin{array}{cc}\hfill i\hfill & \hfill j\hfill \\ \hfill k\hfill & \hfill l\hfill \end{array}\right)\hfill \end{array}\right)$',
    # mathtex doesn't support array
    # 'mmltt24'   : r'$det|\begin{array}{ccccc}\hfill {c}_{0}\hfill & \hfill {c}_{1}\hfill & \hfill {c}_{2}\hfill & \hfill \dots \hfill & \hfill {c}_{n}\hfill \\ \hfill {c}_{1}\hfill & \hfill {c}_{2}\hfill & \hfill {c}_{3}\hfill & \hfill \dots \hfill & \hfill {c}_{n+1}\hfill \\ \hfill {c}_{2}\hfill & \hfill {c}_{3}\hfill & \hfill {c}_{4}\hfill & \hfill \dots \hfill & \hfill {c}_{n+2}\hfill \\ \hfill \u22ee\hfill & \hfill \u22ee\hfill & \hfill \u22ee\hfill & \hfill \hfill & \hfill \u22ee\hfill \\ \hfill {c}_{n}\hfill & \hfill {c}_{n+1}\hfill & \hfill {c}_{n+2}\hfill & \hfill \dots \hfill & \hfill {c}_{2n}\hfill \end{array}|>0$',
    r'${y}_{{x}_{2}}$',
    r'${x}_{92}^{31415}+\pi $',
    r'${x}_{{y}_{b}^{a}}^{{z}_{c}^{d}}$',
    r'${y}_{3}^{\prime \prime \prime }$',
    # End of the MathML torture tests.

    r"$\left( \xi \left( 1 - \xi \right) \right)$",  # Bug 2969451
    r"$\left(2 \, a=b\right)$",  # Sage bug #8125
    r"$? ! &$",  # github issue #466
    None,
    None,
    r"$\left\Vert a \right\Vert \left\vert b \right\vert \left| a \right| \left\| b\right\| \Vert a \Vert \vert b \vert$",
    r'$\mathring{A}  \AA$',
    r'$M \, M \thinspace M \/ M \> M \: M \; M \ M \enspace M \quad M \qquad M \! M$',
    r'$\Cap$ $\Cup$ $\leftharpoonup$ $\barwedge$ $\rightharpoonup$',
    r'$\dotplus$ $\doteq$ $\doteqdot$ $\ddots$',
    r'$xyz^kx_kx^py^{p-2} d_i^jb_jc_kd x^j_i E^0 E^0_u$',  # github issue #4873
    r'${xyz}^k{x}_{k}{x}^{p}{y}^{p-2} {d}_{i}^{j}{b}_{j}{c}_{k}{d} {x}^{j}_{i}{E}^{0}{E}^0_u$',
    r'${\int}_x^x x\oint_x^x x\int_{X}^{X}x\int_x x \int^x x \int_{x} x\int^{x}{\int}_{x} x{\int}^{x}_{x}x$',
    r'testing$^{123}$',
    ' '.join('$\\' + p + '$' for p in sorted(mathtext.Parser._accentprefixed)),
    r'$6-2$; $-2$; $ -2$; ${-2}$; ${  -2}$; $20^{+3}_{-2}$',
    r'$\overline{\omega}^x \frac{1}{2}_0^x$',  # github issue #5444
    r'$,$ $.$ $1{,}234{, }567{ , }890$ and $1,234,567,890$',  # github issue 5799
    r'$\left(X\right)_{a}^{b}$',  # github issue 7615
    r'$\dfrac{\$100.00}{y}$',  # github issue #1888
]
# 'Lightweight' tests test only a single fontset (dejavusans, which is the
# default) and only png outputs, in order to minimize the size of baseline
# images.
lightweight_math_tests = [
    r'$\sqrt[ab]{123}$',  # github issue #8665
    r'$x \overset{f}{\rightarrow} \overset{f}{x} \underset{xx}{ff} \overset{xx}{ff} \underset{f}{x} \underset{f}{\leftarrow} x$',  # github issue #18241
]

digits = "0123456789"
uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lowercase = "abcdefghijklmnopqrstuvwxyz"
uppergreek = ("\\Gamma \\Delta \\Theta \\Lambda \\Xi \\Pi \\Sigma \\Upsilon \\Phi \\Psi "
              "\\Omega")
lowergreek = ("\\alpha \\beta \\gamma \\delta \\epsilon \\zeta \\eta \\theta \\iota "
              "\\lambda \\mu \\nu \\xi \\pi \\kappa \\rho \\sigma \\tau \\upsilon "
              "\\phi \\chi \\psi")
all = [digits, uppercase, lowercase, uppergreek, lowergreek]

# Use stubs to reserve space if tests are removed
# stub should be of the form (None, N) where N is the number of strings that
# used to be tested
# Add new tests at the end.
font_test_specs = [
    ([], all),
    (['mathrm'], all),
    (['mathbf'], all),
    (['mathit'], all),
    (['mathtt'], [digits, uppercase, lowercase]),
    (None, 3),
    (None, 3),
    (None, 3),
    (['mathbb'], [digits, uppercase, lowercase,
                  r'\Gamma \Pi \Sigma \gamma \pi']),
    (['mathrm', 'mathbb'], [digits, uppercase, lowercase,
                            r'\Gamma \Pi \Sigma \gamma \pi']),
    (['mathbf', 'mathbb'], [digits, uppercase, lowercase,
                            r'\Gamma \Pi \Sigma \gamma \pi']),
    (['mathcal'], [uppercase]),
    (['mathfrak'], [uppercase, lowercase]),
    (['mathbf', 'mathfrak'], [uppercase, lowercase]),
    (['mathscr'], [uppercase, lowercase]),
    (['mathsf'], [digits, uppercase, lowercase]),
    (['mathrm', 'mathsf'], [digits, uppercase, lowercase]),
    (['mathbf', 'mathsf'], [digits, uppercase, lowercase])
    ]

font_tests = []
for fonts, chars in font_test_specs:
    if fonts is None:
        font_tests.extend([None] * chars)
    else:
        wrapper = ''.join([
            ' '.join(fonts),
            ' $',
            *(r'\%s{' % font for font in fonts),
            '%s',
            *('}' for font in fonts),
            '$',
        ])
        for set in chars:
            font_tests.append(wrapper % set)


@pytest.fixture
def baseline_images(request, fontset, index, text):
    if text is None:
        pytest.skip("test has been removed")
    return ['%s_%s_%02d' % (request.param, fontset, index)]


@pytest.mark.parametrize(
    'index, text', enumerate(math_tests), ids=range(len(math_tests)))
@pytest.mark.parametrize(
    'fontset', ['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif'])
@pytest.mark.parametrize('baseline_images', ['mathtext'], indirect=True)
@image_comparison(baseline_images=None)
def test_mathtext_rendering(baseline_images, fontset, index, text):
    mpl.rcParams['mathtext.fontset'] = fontset
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text,
             horizontalalignment='center', verticalalignment='center')


@pytest.mark.parametrize('index, text', enumerate(lightweight_math_tests),
                         ids=range(len(lightweight_math_tests)))
@pytest.mark.parametrize('fontset', ['dejavusans'])
@pytest.mark.parametrize('baseline_images', ['mathtext1'], indirect=True)
@image_comparison(baseline_images=None, extensions=['png'])
def test_mathtext_rendering_lightweight(baseline_images, fontset, index, text):
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text, math_fontfamily=fontset,
             horizontalalignment='center', verticalalignment='center')


@pytest.mark.parametrize(
    'index, text', enumerate(font_tests), ids=range(len(font_tests)))
@pytest.mark.parametrize(
    'fontset', ['cm', 'stix', 'stixsans', 'dejavusans', 'dejavuserif'])
@pytest.mark.parametrize('baseline_images', ['mathfont'], indirect=True)
@image_comparison(baseline_images=None, extensions=['png'])
def test_mathfont_rendering(baseline_images, fontset, index, text):
    mpl.rcParams['mathtext.fontset'] = fontset
    fig = plt.figure(figsize=(5.25, 0.75))
    fig.text(0.5, 0.5, text,
             horizontalalignment='center', verticalalignment='center')


def test_fontinfo():
    fontpath = mpl.font_manager.findfont("DejaVu Sans")
    font = mpl.ft2font.FT2Font(fontpath)
    table = font.get_sfnt_table("head")
    assert table['version'] == (1, 0)


@pytest.mark.parametrize(
    'math, msg',
    [
        (r'$\hspace{}$', r'Expected \hspace{n}'),
        (r'$\hspace{foo}$', r'Expected \hspace{n}'),
        (r'$\frac$', r'Expected \frac{num}{den}'),
        (r'$\frac{}{}$', r'Expected \frac{num}{den}'),
        (r'$\binom$', r'Expected \binom{num}{den}'),
        (r'$\binom{}{}$', r'Expected \binom{num}{den}'),
        (r'$\genfrac$',
         r'Expected \genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}'),
        (r'$\genfrac{}{}{}{}{}{}$',
         r'Expected \genfrac{ldelim}{rdelim}{rulesize}{style}{num}{den}'),
        (r'$\sqrt$', r'Expected \sqrt{value}'),
        (r'$\sqrt f$', r'Expected \sqrt{value}'),
        (r'$\overline$', r'Expected \overline{value}'),
        (r'$\overline{}$', r'Expected \overline{value}'),
        (r'$\leftF$', r'Expected a delimiter'),
        (r'$\rightF$', r'Unknown symbol: \rightF'),
        (r'$\left(\right$', r'Expected a delimiter'),
        # PyParsing 2 uses double quotes, PyParsing 3 uses single quotes and an
        # extra backslash.
        (r'$\left($', re.compile(r'Expected ("|\'\\)\\right["\']')),
        (r'$\dfrac$', r'Expected \dfrac{num}{den}'),
        (r'$\dfrac{}{}$', r'Expected \dfrac{num}{den}'),
        (r'$\overset$', r'Expected \overset{body}{annotation}'),
        (r'$\underset$', r'Expected \underset{body}{annotation}'),
    ],
    ids=[
        'hspace without value',
        'hspace with invalid value',
        'frac without parameters',
        'frac with empty parameters',
        'binom without parameters',
        'binom with empty parameters',
        'genfrac without parameters',
        'genfrac with empty parameters',
        'sqrt without parameters',
        'sqrt with invalid value',
        'overline without parameters',
        'overline with empty parameter',
        'left with invalid delimiter',
        'right with invalid delimiter',
        'unclosed parentheses with sizing',
        'unclosed parentheses without sizing',
        'dfrac without parameters',
        'dfrac with empty parameters',
        'overset without parameters',
        'underset without parameters',
    ]
)
def test_mathtext_exceptions(math, msg):
    parser = mathtext.MathTextParser('agg')
    match = re.escape(msg) if isinstance(msg, str) else msg
    with pytest.raises(ValueError, match=match):
        parser.parse(math)


def test_single_minus_sign():
    plt.figure(figsize=(0.3, 0.3))
    plt.text(0.5, 0.5, '$-$')
    plt.gca().spines[:].set_visible(False)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    buff = io.BytesIO()
    plt.savefig(buff, format="rgba", dpi=1000)
    array = np.frombuffer(buff.getvalue(), dtype=np.uint8)

    # If this fails, it would be all white
    assert not np.all(array == 0xff)


@check_figures_equal(extensions=["png"])
def test_spaces(fig_test, fig_ref):
    fig_test.subplots().set_title(r"$1\,2\>3\ 4$")
    fig_ref.subplots().set_title(r"$1\/2\:3~4$")


@check_figures_equal(extensions=["png"])
def test_operator_space(fig_test, fig_ref):
    fig_test.text(0.1, 0.1, r"$\log 6$")
    fig_test.text(0.1, 0.2, r"$\log(6)$")
    fig_test.text(0.1, 0.3, r"$\arcsin 6$")
    fig_test.text(0.1, 0.4, r"$\arcsin|6|$")
    fig_test.text(0.1, 0.5, r"$\operatorname{op} 6$")  # GitHub issue #553
    fig_test.text(0.1, 0.6, r"$\operatorname{op}[6]$")
    fig_test.text(0.1, 0.7, r"$\cos^2$")
    fig_test.text(0.1, 0.8, r"$\log_2$")

    fig_ref.text(0.1, 0.1, r"$\mathrm{log\,}6$")
    fig_ref.text(0.1, 0.2, r"$\mathrm{log}(6)$")
    fig_ref.text(0.1, 0.3, r"$\mathrm{arcsin\,}6$")
    fig_ref.text(0.1, 0.4, r"$\mathrm{arcsin}|6|$")
    fig_ref.text(0.1, 0.5, r"$\mathrm{op\,}6$")
    fig_ref.text(0.1, 0.6, r"$\mathrm{op}[6]$")
    fig_ref.text(0.1, 0.7, r"$\mathrm{cos}^2$")
    fig_ref.text(0.1, 0.8, r"$\mathrm{log}_2$")


def test_mathtext_fallback_valid():
    for fallback in ['cm', 'stix', 'stixsans', 'None']:
        mpl.rcParams['mathtext.fallback'] = fallback


def test_mathtext_fallback_invalid():
    for fallback in ['abc', '']:
        with pytest.raises(ValueError, match="not a valid fallback font name"):
            mpl.rcParams['mathtext.fallback'] = fallback


@pytest.mark.parametrize(
    "fallback,fontlist",
    [("cm", ['DejaVu Sans', 'mpltest', 'STIXGeneral', 'cmr10', 'STIXGeneral']),
     ("stix", ['DejaVu Sans', 'mpltest', 'STIXGeneral'])])
def test_mathtext_fallback(fallback, fontlist):
    mpl.font_manager.fontManager.addfont(
        str(Path(__file__).resolve().parent / 'mpltest.ttf'))
    mpl.rcParams["svg.fonttype"] = 'none'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'mpltest'
    mpl.rcParams['mathtext.it'] = 'mpltest:italic'
    mpl.rcParams['mathtext.bf'] = 'mpltest:bold'
    mpl.rcParams['mathtext.fallback'] = fallback

    test_str = r'a$A\AA\breve\gimel$'

    buff = io.BytesIO()
    fig, ax = plt.subplots()
    fig.text(.5, .5, test_str, fontsize=40, ha='center')
    fig.savefig(buff, format="svg")
    tspans = (ET.fromstring(buff.getvalue())
              .findall(".//{http://www.w3.org/2000/svg}tspan[@style]"))
    # Getting the last element of the style attrib is a close enough
    # approximation for parsing the font property.
    char_fonts = [shlex.split(tspan.attrib["style"])[-1] for tspan in tspans]
    assert char_fonts == fontlist
    mpl.font_manager.fontManager.ttflist.pop()


def test_math_to_image(tmpdir):
    mathtext.math_to_image('$x^2$', str(tmpdir.join('example.png')))
    mathtext.math_to_image('$x^2$', io.BytesIO())


def test_mathtext_to_png(tmpdir):
    with _api.suppress_matplotlib_deprecation_warning():
        mt = mathtext.MathTextParser('bitmap')
        mt.to_png(str(tmpdir.join('example.png')), '$x^2$')
        mt.to_png(io.BytesIO(), '$x^2$')


@image_comparison(baseline_images=['math_fontfamily_image.png'],
                  savefig_kwarg={'dpi': 40})
def test_math_fontfamily():
    fig = plt.figure(figsize=(10, 3))
    fig.text(0.2, 0.7, r"$This\ text\ should\ have\ one\ font$",
             size=24, math_fontfamily='dejavusans')
    fig.text(0.2, 0.3, r"$This\ text\ should\ have\ another$",
             size=24, math_fontfamily='stix')


def test_default_math_fontfamily():
    mpl.rcParams['mathtext.fontset'] = 'cm'
    test_str = r'abc$abc\alpha$'
    fig, ax = plt.subplots()

    text1 = fig.text(0.1, 0.1, test_str, font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'cm'
    text2 = fig.text(0.2, 0.2, test_str, fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'cm'

    fig.draw_without_rendering()


def test_argument_order():
    mpl.rcParams['mathtext.fontset'] = 'cm'
    test_str = r'abc$abc\alpha$'
    fig, ax = plt.subplots()

    text1 = fig.text(0.1, 0.1, test_str,
                     math_fontfamily='dejavusans', font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'dejavusans'
    text2 = fig.text(0.2, 0.2, test_str,
                     math_fontfamily='dejavusans', fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'dejavusans'
    text3 = fig.text(0.3, 0.3, test_str,
                     font='Arial', math_fontfamily='dejavusans')
    prop3 = text3.get_fontproperties()
    assert prop3.get_math_fontfamily() == 'dejavusans'
    text4 = fig.text(0.4, 0.4, test_str,
                     fontproperties='Arial', math_fontfamily='dejavusans')
    prop4 = text4.get_fontproperties()
    assert prop4.get_math_fontfamily() == 'dejavusans'

    fig.draw_without_rendering()


def test_mathtext_cmr10_minus_sign():
    # cmr10 does not contain a minus sign and used to issue a warning
    # RuntimeWarning: Glyph 8722 missing from current font.
    mpl.rcParams['font.family'] = 'cmr10'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    fig, ax = plt.subplots()
    ax.plot(range(-1, 1), range(-1, 1))
    # draw to make sure we have no warnings
    fig.canvas.draw()
