(() => {
  const trigger = document.getElementById('build-skills-trigger');
  const dialog = document.getElementById('build-skills-dialog');
  if (!trigger || !dialog) return;

  const title = document.getElementById('build-skills-title');
  const closeButton = dialog.querySelector('.build-skills-close');
  const backButton = dialog.querySelector('.build-skills-back');
  const continueLink = dialog.querySelector('.build-skills-continue');
  const content = dialog.querySelector('.build-skills-content');
  const tabs = [...dialog.querySelectorAll('[role="tab"]')];
  const labels = [
    { lang: 'en', title: 'Welcome to ScienceCUA Build Skills', close: 'Close introduction', back: 'Back', next: 'Continue' },
    { lang: 'zh-CN', title: '欢迎体验 ScienceCUA Build Skills', close: '关闭说明', back: '返回', next: '继续' },
  ];

  const activateLanguage = (selected) => {
    tabs.forEach((tab) => {
      const active = tab === selected;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
    });
    const translation = labels[tabs.indexOf(selected)];
    dialog.lang = translation.lang;
    title.textContent = translation.title;
    closeButton.setAttribute('aria-label', translation.close);
    backButton.textContent = translation.back;
    continueLink.querySelector('span').textContent = translation.next;
    content.scrollTop = 0;
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activateLanguage(tab));
    tab.addEventListener('keydown', (event) => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      activateLanguage(tabs[next]);
      tabs[next].focus();
    });
  });

  trigger.addEventListener('click', () => {
    if (dialog.open) return;
    dialog.showModal();
    trigger.setAttribute('aria-expanded', 'true');
    document.documentElement.classList.add('build-skills-open');
    content.scrollTop = 0;
  });

  closeButton.addEventListener('click', () => dialog.close());
  backButton.addEventListener('click', () => dialog.close());
  continueLink.addEventListener('click', () => dialog.close());

  dialog.addEventListener('click', (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
      dialog.close();
    }
  });

  dialog.addEventListener('close', () => {
    trigger.setAttribute('aria-expanded', 'false');
    document.documentElement.classList.remove('build-skills-open');
    trigger.focus({ preventScroll: true });
  });

  const preferredLanguage = navigator.language || '';
  activateLanguage(tabs[preferredLanguage.toLowerCase().startsWith('zh') ? 1 : 0]);
})();
