import re
import pandas as pd


CONTENT = '''
Threshold (dB SPL): {threshold:.2f}
Frequency (kHz): {frequency:.2f}
Filter history (zpk format): {filter_history}
NOTE: Negative latencies indicate no peak
{columns}
{spreadsheet}
'''.strip()


def get_save_filename(self, filename, frequency):
    raise NotImplementedError
    # # Round frequency to nearest 8 places to minimize floating-point
    # # errors.
    # user_name = self._user + '-' if self._user else ''
    # frequency = round(frequency * 1e-3, 8)
    # save_filename = self.filename_template.format(
    #     filename=filename.with_suffix(''),
    #     frequency=frequency,
    #     user=user_name)
    # return Path(save_filename)

def load_analysis(fname):
    th_match = re.compile('Threshold \(dB SPL\): ([\w.]+)')
    freq_match = re.compile('Frequency \(kHz\): ([\d.]+)')
    with open(fname) as fh:
        text = fh.readline()
        th = th_match.search(text).group(1)
        th = None if th == 'None' else float(th)
        text = fh.readline()
        freq = float(freq_match.search(text).group(1))

        for line in fh:
            if line.startswith('NOTE'):
                break
        data = pd.io.parsers.read_csv(fh, sep='\t', index_col='Level')
    return (freq, th, data)


def filter_string(waveform):
    if getattr(waveform, '_zpk', None) is None:
        return 'No filtering'
    t = 'Pass %d -- z: %r, p: %r, k: %r'
    filt = [t % (i, z, p, k) for i, (z, p, k) in enumerate(waveform._zpk)]
    return '\n' + '\n'.join(filt)

def save(model):
    raise NotImplementedError
    # # Assume that all waveforms were filtered identically
    # filter_history = filter_string(model.waveforms[-1])
    #
    # # Generate list of columns
    # columns = ['Level', '1msec Avg', '1msec StDev']
    # point_keys = sorted(model.waveforms[0].points)
    # for point_number, point_type in point_keys:
    #     point_type_code = 'P' if point_type == Point.PEAK else 'N'
    #     for measure in ('Latency', 'Amplitude'):
    #         columns.append(f'{point_type_code}{point_number} {measure}')
    #
    # columns = '\t'.join(columns)
    # spreadsheet = '\n'.join(waveform_string(w) \
    #                         for w in reversed(model.waveforms))
    # content = CONTENT.format(threshold=model.threshold,
    #                          frequency=model.freq * 1e-3,
    #                          filter_history=filter_history,
    #                          columns=columns,
    #                          spreadsheet=spreadsheet)
    #
    # filename = get_save_filename(model.filename, model.freq)
    # with open(filename, 'w') as fh:
    #     fh.writelines(content)