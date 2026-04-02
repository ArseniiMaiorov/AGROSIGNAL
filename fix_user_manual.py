import re

path = '/home/arsenii-maiorov/Documents/SUAI/Диплом/РП/docs/user_manual.tex'
try:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

replacements = {
    # Bootstrap
    r'bootstrap-проверок': r'стартовых проверок',
    r'bootstrap-доступа': r'базового стартового доступа',
    r'bootstrap-подсказка': r'стартовая подсказка',
    r'Bootstrap-подсказка': r'Стартовая подсказка',
    r'bootstrap-учётной': r'стартовой учётной',
    
    # Preflight
    r'Preflight-проверку': r'предварительную проверку',
    r'Preflight-проверки': r'предварительной проверки',
    r'\(Preflight\)': r'(предварительной проверки)',
    r'Preflight завершается': r'предварительная проверка завершается',
    r'Preflight отклонён': r'Предварительная проверка отклонена',
    
    # Roles
    r'backend-подсистем': r'подсистем серверной части',
    r'backend': r'серверной части',
    r'frontend': r'клиентской части',
    r'воркера': r'фонового обработчика',
    r'воркер': r'фоновый обработчик',
    r'Celery Worker': r'Фоновый обработчик Celery',
    
    # Profiles
    r'\\textit\{Fast\} / \\textit\{Standard\} / \\textit\{Quality\}': r'\\textit{Быстрый} / \\textit{Стандартный} / \\textit{Качественный}',
    r'& \\textbf\{Fast\} & \\textbf\{Standard\} & \\textbf\{Quality\} \\\\': r'& \\textbf{Быстрый} & \\textbf{Стандартный} & \\textbf{Качественный} \\\\',
    r'Пресет Fast': r'Пресет «Быстрый»',
    r'Пресет Standard': r'Пресет «Стандартный»',
    r'Пресет Quality': r'Пресет «Качественный»',
    r'\(Fast\)': r'(Быстрый)',
    r'\(Standard\)': r'(Стандартный)',
    r'\(Quality\)': r'(Качественный)',
    r'пресета \\textit\{Fast\}': r'пресета \\textit{Быстрый}',
    r'\\textit\{Fast\}, \\textit\{Standard\} и \\textit\{Quality\}': r'\\textit{Быстрый}, \\textit{Стандартный} и \\textit{Качественный}',
    r'Quality-пресет': r'пресет «Качественный»',
    
    # UI Elements and English Jargon
    r'\\uikey\{Sign In\}': r'\\uikey{Войти}',
    r'\\uibtn\{Sign In\}': r'\\uibtn{Войти}',
    r'\\uibtn\{Enter\}': r'\\uibtn{Ввод}',
    r'\\textbf\{Email\}': r'\\textbf{Электронная почта}',
    r'\\textbf\{Password\}': r'\\textbf{Пароль}',
    r'\\textbf\{Organization Slug\}': r'\\textbf{Идентификатор организации}',
    r'Organization Slug': r'Идентификатор организации',
    r'\\texttt\{default-\\allowbreak organization\}': r'\\texttt{default-\\allowbreak organization} (стандартная организация)',
    r'\\textit\{auto\}': r'\\textit{автоматический}',
    r'\\textit\{manual\}': r'\\textit{ручной}',
    r'\\texttt\{manual\}': r'\\texttt{ручной}',
    r'\(drivers\)': r'',
    r'\(quality tier\)': r'',
    r'\(canopy cover\)': r'',
    r'harvest index': r'индекс урожая',
    r'\(watershed\)': r'(алгоритм водораздела)',
    r'HiFi': r'высокой точности',
    r'Rate limit exceeded': r'Лимит запросов превышен',
    r'RAM': r'ОЗУ',
    
    r'FreshnessBadge': r'Метка актуальности',
    r'\\textbf\{Lat\}': r'\\textbf{Широта}',
    r'\\textbf\{Lon\}': r'\\textbf{Долгота}',
    r'\\uikey\{Lat\}': r'\\uikey{Широта}',
    r'\\uikey\{Lon\}': r'\\uikey{Долгота}',
    
    # Model Stages inside table
    r'\\texttt\{queued\}': r'\\texttt{в очереди}',
    r'\\texttt\{fetch\}': r'\\texttt{загрузка}',
    r'\\texttt\{tiling\}': r'\\texttt{разбиение на тайлы}',
    r'\\makecell\[l\]\{\\texttt\{date\_\\\}\\\\\\\\\n\\texttt\{selection\}\}': r'\\makecell[l]{\\texttt{отбор\_}\\\\\\\\\\texttt{дат}}',
    r'\\makecell\[l\]\{\\texttt\{candidate\_\\\}\\\\\\\\\n\\texttt\{postprocess\}\}': r'\\makecell[l]{\\texttt{постобработка\_}\\\\\\\\\\texttt{кандидатов}}',
    r'\\makecell\[l\]\{\\texttt\{model\_\\\}\\\\\\\\\n\\texttt\{inference\}\}': r'\\makecell[l]{\\texttt{применение\_}\\\\\\\\\\texttt{модели}}',
    r'\\texttt\{segmentation\}': r'\\texttt{сегментация}',
    r'\\makecell\[l\]\{\\texttt\{boundary\_\\\}\\\\\\\\\n\\texttt\{refine\}\}': r'\\makecell[l]{\\texttt{уточнение\_}\\\\\\\\\\texttt{границ}}',
    r'\\texttt\{sam_refine\}': r'\\texttt{уточнение\_SAM}',
    r'\\makecell\[l\]\{\\texttt\{tile\_\\\}\\\\\\\\\n\\texttt\{finalize\}\}': r'\\makecell[l]{\\texttt{финализация\_}\\\\\\\\\\texttt{тайла}}',
    r'\\texttt\{merge\}': r'\\texttt{слияние}',
    r'\\makecell\[l\]\{\\texttt\{object\_\\\}\\\\\\\\\n\\texttt\{classifier\}\}': r'\\makecell[l]{\\texttt{классификатор\_}\\\\\\\\\\texttt{объектов}}',
    r'\\texttt\{db_insert\}': r'\\texttt{запись\_в\_БД}',
    r'\\texttt\{done\}': r'\\texttt{успешно}',
    r'\\texttt\{failed\}': r'\\texttt{сбой}',
}

# The line breaks might be different, let's use exact simple replaces for the multi-line makecell
text = text.replace(r'\makecell[l]{\texttt{date\_}\\', r'\makecell[l]{\texttt{отбор\_}\\')
text = text.replace(r'\texttt{selection}} & Отбор снимков по облачности и качеству \\', r'\texttt{дат}} & Отбор снимков по облачности и качеству \\')

text = text.replace(r'\makecell[l]{\texttt{candidate\_}\\', r'\makecell[l]{\texttt{постобработка\_}\\')
text = text.replace(r'\texttt{postprocess}} & Эвристическое построение кандидатов границ \\', r'\texttt{кандидатов}} & Эвристическое построение кандидатов границ \\')

text = text.replace(r'\makecell[l]{\texttt{model\_}\\', r'\makecell[l]{\texttt{применение\_}\\')
text = text.replace(r'\texttt{inference}} & Инференс нейросети BoundaryUNet~v2 \\', r'\texttt{модели}} & Применение нейросети BoundaryUNet~v2 \\')

text = text.replace(r'\makecell[l]{\texttt{boundary\_}\\', r'\makecell[l]{\texttt{уточнение\_}\\')
text = text.replace(r'\texttt{refine}} & Уточнение границ полей \\', r'\texttt{границ}} & Уточнение границ полей \\')

text = text.replace(r'\makecell[l]{\texttt{tile\_}\\', r'\makecell[l]{\texttt{финализация\_}\\')
text = text.replace(r'\texttt{finalize}} & Сборка результатов тайлов \\', r'\texttt{тайла}} & Сборка результатов тайлов \\')

text = text.replace(r'\makecell[l]{\texttt{object\_}\\', r'\makecell[l]{\texttt{классификатор\_}\\')
text = text.replace(r'\texttt{classifier}} & Фильтрация ложных срабатываний классификатором \\', r'\texttt{объектов}} & Фильтрация ложных срабатываний классификатором \\')

for k, v in replacements.items():
    text = re.sub(k, v, text)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("ALL_DONE")
