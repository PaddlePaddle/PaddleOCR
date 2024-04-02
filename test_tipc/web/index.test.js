const expectData = require('./expect.json');

describe('e2e test ocr model', () => {

    beforeAll(async () => {
        await page.goto(PATH);
    });

    it('ocr infer and diff test', async () => {
        page.on('console', msg => console.log('PAGE LOG:', msg.text()));

        const text = await page.evaluate(async () => {
            const $ocr = document.querySelector('#ocr');
            const ocr = paddlejs['ocr'];
            await ocr.init('./models/ch_PP-OCRv2_det_infer', './models/ch_PP-OCRv2_rec_infer');
            const res = await ocr.recognize($ocr);
            return res.text;
        });
        // 模型文字识别结果与预期结果diff的字符数
        let diffNum = 0;
        // 文本框字符串相似度
        let similarity = 0;
        // 预期字符diff数
        const expectedDiffNum = 10;
        // 预期文本框字符串相似度
        const expecteSimilarity = 0.9;
        // 预期文本内容
        const expectResult = expectData.text;

        expectResult && expectResult.forEach((item, index) => {
            const word = text[index];
            // 逐字符对比
            for(let i = 0; i < item.length; i++) {
                if (item[i] !== word[i]) {
                    console.log('expect: ', item[i], ' word: ', word[i]);
                    diffNum++;
                }
            }
            // 文本框字符串相似度对比
            const s = similar(item, word);
            similarity += s;
        });

        similarity = similarity / expectResult.length;

        expect(diffNum).toBeLessThanOrEqual(expectedDiffNum);

        expect(similarity).toBeGreaterThanOrEqual(expecteSimilarity);

        function similar(string, expect) {
            if (!string || !expect) {
                return 0;
            }
            const length = string.length > expect.length ? string.length : expect.length;
            const n = string.length;
            const m = expect.length;
            let data = [];
            const min = (a, b, c) => {
                return a < b ? (a < c ? a : c) : (b < c ? b : c);
            };
            let i, j, si, ej, cost;
            if (n === 0) return m;
            if (m === 0) return n;
            for (i = 0; i <= n; i++) {
                data[i] = [];
                [i][0] = i
            }
            for (j = 0; j <= m; j++) {
                data[0][j] = j;
            }
            for (i = 1; i <= n; i++) {
                si = string.charAt(i - 1);
                for (j = 1; j <= m; j++) {
                    ej = expect.charAt(j - 1);
                    cost = si === ej ? 0 : 1;
                    data[i][j] = min(data[i - 1][j] + 1, data[i][j - 1] + 1, data[i - 1][j - 1] + cost);
                }
            }
            return (1 - data[n][m] / length);
        }
    });
});
