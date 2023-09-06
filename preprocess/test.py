code='''
static av_cold int roq_dpcm_encode_init(AVCodecContext *avctx)
{
    ROQDPCMContext *context ;
    context = avctx->priv_data;
    if (avctx->channels > 2) {
        av_log(avctx, AV_LOG_ERROR, "Audio must be mono or stereo
");
        return -1;
    }
    if (avctx->sample_rate != 22050) {
        av_log(avctx, AV_LOG_ERROR, "Audio must be 22050 Hz
");
        return -1;
    }
    if (avctx->sample_fmt != AV_SAMPLE_FMT_S16) {
        av_log(avctx, AV_LOG_ERROR, "Audio must be signed 16-bit
");
        return -1;
    }
    avctx->frame_size = ROQ_FIRST_FRAME_SIZE;
    context->lastSample[0] = context->lastSample[1] = 0;
    avctx->coded_frame= avcodec_alloc_frame();

    return 0;
}'''

def find_quotes_index(string):
    stack = []
    for index, char in enumerate(string):
        if char in ['\'', '\"'] and len(stack) == 0:
            stack.append((char, index))
        elif char == '\'' and stack[-1][0] == '\'' or char == '\"' and stack[-1][0] == '\"':
            _, l = stack.pop()
            if len(stack) == 0:
                return l, index + 1
    return -1, -1

def replace_n(code): # 将双引号里面的\n改为\\n
    i = 0
    while i < len(code):
        l, r = find_quotes_index(code[i:])
        if l == -1:
            break
        code = code[:i + l] + code[i + l:i + r].replace('\n','\\n') + code[i + r:]
        i += r + 1
    return code

print(code)
print(replace_n(code))